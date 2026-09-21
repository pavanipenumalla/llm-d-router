/*
Copyright 2025 The Kubernetes Authors.
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package prefixcacheaffinity provides a probabilistic filter that narrows
// candidates to "sticky" endpoints (those with high prefix cache scores).
// Can be instantiated multiple times with different thresholds (e.g., 0.99
// for global gate, 0.80 for within-tier gate).
package prefixcacheaffinity

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"

	"go.opentelemetry.io/otel/trace"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionstate"
	schedplugins "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling"
)

const (
	PluginType = "prefix-cache-affinity-filter"
)

var (
	_ fwksched.Filter           = &Plugin{}
	_ requestcontrol.PreRequest = &Plugin{}
)

// TTFTSource selects the per-endpoint TTFT signal used by the load gate. The
// choice also determines which producer attribute the filter consumes.
type TTFTSource string

const (
	// TTFTSourceLatencyPredictor reads predicted TTFT from LatencyPredictionInfo
	// (produced by the predicted-latency-producer).
	TTFTSourceLatencyPredictor TTFTSource = "latencyPredictor"
	// TTFTSourcePrefillThroughput estimates TTFT from in-flight tokens and
	// PeakPrefillThroughput, reading InFlightLoad (produced by the
	// in-flight-load-producer).
	TTFTSourcePrefillThroughput TTFTSource = "prefillThroughput"
)

// PenaltySource selects how the filter decides whether load justifies giving up
// cached context.
type PenaltySource string

const (
	// PenaltySourceConstant compares the TTFT gap between the best sticky and
	// best non-sticky endpoint against MaxTTFTPenaltyMs, keeping all endpoints
	// when the gap is wider.
	PenaltySourceConstant PenaltySource = "constant"
	// PenaltySourceSessionCost scores every candidate on the prefill latency a
	// session expects to pay over its remaining turns, and narrows to whichever
	// of the two sets is cheaper. Reads UncachedRequestTokens (produced by the
	// in-flight-load-producer) in addition to the TTFT source.
	PenaltySourceSessionCost PenaltySource = "sessionCost"
)

type Config struct {
	// AffinityThreshold is the prefix cache score threshold. Endpoints with
	// score >= this value are considered "sticky" (prompt is cached). Default: 0.80.
	AffinityThreshold float64 `json:"affinityThreshold,omitempty"`

	// ExplorationProbability is the probability of skipping the gate entirely,
	// keeping all endpoints for exploration. Range: [0, 1]. Default: 0.
	ExplorationProbability float64 `json:"explorationProbability,omitempty"`

	// MaxTTFTPenaltyMs is the max TTFT penalty (ms) before breaking stickiness.
	// If the best sticky endpoint's TTFT exceeds the best non-sticky endpoint's
	// TTFT by more than this value, all endpoints are kept. Set to 0 to always
	// stick. Default: 18000.
	MaxTTFTPenaltyMs float64 `json:"maxTTFTPenaltyMs,omitempty"`

	// TTFTSource selects where the load gate reads per-endpoint TTFT from.
	// TTFTSourcePrefillThroughput (default) estimates it from in-flight tokens and
	// PeakPrefillThroughput; TTFTSourceLatencyPredictor reads predicted TTFT from
	// the latency predictor.
	TTFTSource TTFTSource `json:"ttftSource,omitempty"`

	// PeakPrefillThroughput is the peak prefill throughput in tokens/sec, used to
	// estimate TTFT from in-flight tokens when TTFTSource is prefillThroughput:
	//   TTFT_ms = inFlightTokens / PeakPrefillThroughput * 1000
	// (tokens / (tokens/sec) * 1000 = ms). Default: 15928.
	//
	// PenaltySourceSessionCost also converts a candidate's uncached tokens to
	// milliseconds with it, on either TTFTSource, so that mode requires a
	// non-zero value.
	PeakPrefillThroughput float64 `json:"peakPrefillThroughput,omitempty"`

	// PenaltySource selects how stickiness is weighed against load.
	// PenaltySourceConstant (default) preserves the MaxTTFTPenaltyMs gate.
	// Default: constant.
	PenaltySource PenaltySource `json:"penaltySource,omitempty"`

	PrefixMatchInfoProducerName       string `json:"prefixMatchInfoProducerName,omitempty"`
	LatencyPredictionInfoProducerName string `json:"latencyPredictionInfoProducerName,omitempty"`
	InFlightLoadProducerName          string `json:"inFlightLoadProducerName,omitempty"`
}

var DefaultConfig = Config{
	AffinityThreshold:      0.80,
	ExplorationProbability: 0,
	MaxTTFTPenaltyMs:       18000,
	TTFTSource:             TTFTSourcePrefillThroughput,
	PenaltySource:          PenaltySourceConstant,

	// Calibrated for Qwen 32B on 2x H100 80GB (TP=2), vLLM 0.19; see README.
	PeakPrefillThroughput: 15928,
}

type Plugin struct {
	typedName                    fwkplugin.TypedName
	config                       Config
	prefixMatchDataKey           fwkplugin.DataKey
	latencyPredictionInfoDataKey fwkplugin.DataKey
	inFlightLoadDataKey          fwkplugin.DataKey
	uncachedRequestTokensDataKey fwkplugin.DataKey

	// survival is non-nil only under PenaltySourceSessionCost. Each configured
	// instance keeps its own, so two instances counting the same request each
	// record it once in their own table rather than twice in a shared one.
	survival *turnSurvival
}

func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config: %w", err)
		}
	}
	if err := config.validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
		if config.usesSessionCost() && config.MaxTTFTPenaltyMs > 0 {
			log.FromContext(handle.Context()).Info(
				"maxTTFTPenaltyMs is not used when penaltySource is sessionCost",
				"plugin", name, "maxTTFTPenaltyMs", config.MaxTTFTPenaltyMs)
		}
	}
	plugin := &Plugin{
		typedName:                    fwkplugin.TypedName{Type: PluginType, Name: name},
		config:                       config,
		prefixMatchDataKey:           attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(config.PrefixMatchInfoProducerName),
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(config.LatencyPredictionInfoProducerName),
		inFlightLoadDataKey:          attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
		uncachedRequestTokensDataKey: attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
	}
	if config.usesSessionCost() {
		plugin.survival = newTurnSurvival()
	}
	return plugin, nil
}

func (c *Config) validate() error {
	if c.AffinityThreshold < 0 || c.AffinityThreshold > 1.0 {
		return fmt.Errorf("affinityThreshold must be in [0, 1], got %f", c.AffinityThreshold)
	}
	if c.ExplorationProbability < 0 || c.ExplorationProbability > 1.0 {
		return fmt.Errorf("explorationProbability must be in [0, 1], got %f", c.ExplorationProbability)
	}
	if c.MaxTTFTPenaltyMs < 0 {
		return fmt.Errorf("maxTTFTPenaltyMs must be >= 0, got %f", c.MaxTTFTPenaltyMs)
	}
	if c.PeakPrefillThroughput < 0 {
		return fmt.Errorf("peakPrefillThroughput must be >= 0, got %f", c.PeakPrefillThroughput)
	}
	switch c.TTFTSource {
	case TTFTSourceLatencyPredictor, TTFTSourcePrefillThroughput:
	default:
		return fmt.Errorf("ttftSource must be %q or %q, got %q", TTFTSourceLatencyPredictor, TTFTSourcePrefillThroughput, c.TTFTSource)
	}
	switch c.PenaltySource {
	case PenaltySourceConstant, PenaltySourceSessionCost:
	default:
		return fmt.Errorf("penaltySource must be %q or %q, got %q", PenaltySourceConstant, PenaltySourceSessionCost, c.PenaltySource)
	}
	if !c.usesLatencyPredictor() && c.MaxTTFTPenaltyMs > 0 && c.PeakPrefillThroughput == 0 {
		return errors.New("peakPrefillThroughput must be > 0 when ttftSource is prefillThroughput")
	}
	// sessionCost converts uncached tokens to milliseconds on either TTFT source,
	// so the throughput figure is load-bearing even on the predictor path, where
	// the constant gate permits zero.
	if c.usesSessionCost() && c.PeakPrefillThroughput == 0 {
		return errors.New("peakPrefillThroughput must be > 0 when penaltySource is sessionCost")
	}
	return nil
}

// usesLatencyPredictor reports whether the load gate sources TTFT from the
// latency predictor. Throughput is the default; only an explicit
// latencyPredictor selects the predictor.
func (c *Config) usesLatencyPredictor() bool {
	return c.TTFTSource == TTFTSourceLatencyPredictor
}

// usesSessionCost reports whether the filter narrows by expected session cost
// rather than by the constant TTFT penalty. Constant is the default; only an
// explicit sessionCost selects the cost function.
func (c *Config) usesSessionCost() bool {
	return c.PenaltySource == PenaltySourceSessionCost
}

func (p *Plugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *Plugin) Filter(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx)

	_, span := tracing.Tracer(schedplugins.TracerScope).Start(ctx, "filter_prefix_cache_affinity",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	span.SetAttributes(
		semconv.LLMDEPPFilterCandidateEndpoints(len(endpoints)),
		semconv.LLMDEPPFilterAffinityThreshold(p.config.AffinityThreshold),
	)
	if request != nil {
		if request.TargetModel != "" {
			span.SetAttributes(semconv.GenAIRequestModel(request.TargetModel))
		}
		if request.RequestID != "" {
			span.SetAttributes(semconv.GenAIRequestID(request.RequestID))
		}
	}

	if len(endpoints) <= 1 || p.config.AffinityThreshold <= 0 {
		recordDecision(p.typedName.Name, outcomeNotApplicable)
		span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeNotApplicable))
		return endpoints
	}

	// Exploration: skip the gate with configured probability.
	if rand.Float64() < p.config.ExplorationProbability {
		logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: exploration skip, keeping all",
			"affinityThreshold", p.config.AffinityThreshold, "total", len(endpoints))
		recordDecision(p.typedName.Name, outcomeExploration)
		span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeExploration))
		return endpoints
	}

	// Find sticky and non-sticky endpoints.
	var sticky, nonSticky []fwksched.Endpoint
	for _, ep := range endpoints {
		if p.prefixCacheScore(ep) >= p.config.AffinityThreshold {
			sticky = append(sticky, ep)
		} else {
			nonSticky = append(nonSticky, ep)
		}
	}

	span.SetAttributes(semconv.LLMDEPPFilterStickyEndpoints(len(sticky)))

	// No sticky endpoints found, keep all.
	if len(sticky) == 0 {
		logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: no sticky endpoints",
			"affinityThreshold", p.config.AffinityThreshold, "total", len(endpoints))
		recordDecision(p.typedName.Name, outcomeNoMatch)
		span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeNoMatch))
		return endpoints
	}

	// Session cost gate: narrow to whichever set is cheaper over the session's
	// remaining turns. Unlike the constant gate below this discards the losing
	// set, so the comparison decides where the request goes rather than leaving
	// it to the scorers.
	if p.config.usesSessionCost() && len(nonSticky) > 0 {
		remainingTurns := p.remainingTurns(request)
		stickyCost := p.bestCost(sticky, remainingTurns)
		nonStickyCost := p.bestCost(nonSticky, remainingTurns)
		if v := logger.V(logutil.DEBUG); v.Enabled() {
			v.Info("PrefixCacheAffinityFilter: session cost gate",
				"stickyCost", stickyCost, "nonStickyCost", nonStickyCost,
				"remainingTurns", remainingTurns)
		}
		if nonStickyCost < stickyCost {
			recordDecision(p.typedName.Name, outcomeSessionCostMigrate)
			span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeSessionCostMigrate))
			return nonSticky
		}
		recordDecision(p.typedName.Name, outcomeSessionCostStay)
		span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeSessionCostStay))
		return sticky
	}

	// TTFT load gate: break stickiness if sticky endpoints are too slow.
	if p.config.MaxTTFTPenaltyMs > 0 && len(nonSticky) > 0 {
		bestStickyTTFT := p.bestTTFT(sticky)
		bestNonStickyTTFT := p.bestTTFT(nonSticky)
		penalty := bestStickyTTFT - bestNonStickyTTFT
		span.SetAttributes(semconv.LLMDEPPFilterTTFTPenaltyMs(penalty))
		if penalty > p.config.MaxTTFTPenaltyMs {
			logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: TTFT load gate broken",
				"bestStickyTTFT", bestStickyTTFT, "bestNonStickyTTFT", bestNonStickyTTFT,
				"penalty", penalty, "maxPenalty", p.config.MaxTTFTPenaltyMs)
			recordDecision(p.typedName.Name, outcomeLoadOverride)
			span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeLoadOverride))
			return endpoints
		}
	}

	logger.V(logutil.DEBUG).Info("PrefixCacheAffinityFilter: narrowed to sticky",
		"affinityThreshold", p.config.AffinityThreshold, "sticky", len(sticky), "total", len(endpoints))
	recordDecision(p.typedName.Name, outcomeSticky)
	span.SetAttributes(semconv.LLMDEPPFilterDecision(outcomeSticky))
	return sticky
}

func (p *Plugin) Consumes() fwkplugin.DataDependencies {
	required := map[fwkplugin.DataKey]any{
		p.prefixMatchDataKey: attrprefix.PrefixCacheMatchInfo{},
	}
	// sessionCost reads the TTFT source on every call, so the dependency does not
	// hinge on MaxTTFTPenaltyMs, which that mode ignores.
	if p.config.MaxTTFTPenaltyMs > 0 || p.config.usesSessionCost() {
		if p.config.usesLatencyPredictor() {
			required[p.latencyPredictionInfoDataKey] = attrlatency.LatencyPredictionInfo{}
		} else {
			required[p.inFlightLoadDataKey] = attrconcurrency.InFlightLoad{}
		}
	}
	if p.config.usesSessionCost() {
		required[p.uncachedRequestTokensDataKey] = attrconcurrency.UncachedRequestTokens{}
		required[sessionstate.SessionStateDataKey] = sessionstate.SessionState{}
	}
	return fwkplugin.DataDependencies{Required: required}
}

func (p *Plugin) prefixCacheScore(ep fwksched.Endpoint) float64 {
	if raw, ok := ep.Get(p.prefixMatchDataKey); ok {
		info := raw.(*attrprefix.PrefixCacheMatchInfo)
		if info.TotalBlocks() > 0 {
			score := float64(info.MatchBlocks()) / float64(info.TotalBlocks())
			if !math.IsNaN(score) {
				return score
			}
		}
	}
	return 0
}

// bestTTFT returns the lowest per-endpoint TTFT (ms) across endpoints.
func (p *Plugin) bestTTFT(endpoints []fwksched.Endpoint) float64 {
	best := math.MaxFloat64
	for _, ep := range endpoints {
		if ttft := p.endpointTTFT(ep); ttft < best {
			best = ttft
		}
	}
	return best
}

// endpointTTFT returns the predicted TTFT (ms) for an endpoint, either from the
// latency predictor or estimated from in-flight tokens and peak prefill
// throughput. Endpoints missing the required attribute contribute no signal:
// MaxFloat64 on the predictor path (never the fastest), 0 in-flight tokens on
// the throughput path (no observed load).
func (p *Plugin) endpointTTFT(ep fwksched.Endpoint) float64 {
	if p.config.usesLatencyPredictor() {
		if raw, ok := ep.Get(p.latencyPredictionInfoDataKey); ok {
			info := raw.(*attrlatency.LatencyPredictionInfo)
			return info.TTFT()
		}
		return math.MaxFloat64
	}
	return float64(p.inFlightTokens(ep)) / p.config.PeakPrefillThroughput * 1000
}

// bestCost returns the lowest session cost (ms) across endpoints.
func (p *Plugin) bestCost(endpoints []fwksched.Endpoint, remainingTurns float64) float64 {
	best := math.MaxFloat64
	for _, ep := range endpoints {
		if cost := p.sessionCost(ep, remainingTurns); cost < best {
			best = cost
		}
	}
	return best
}

// sessionCost returns the prefill latency (ms) a session expects to pay on an
// endpoint over its remaining turns: the prefill it owes here, paid once, plus
// this endpoint's queueing delay on every remaining turn.
func (p *Plugin) sessionCost(ep fwksched.Endpoint, remainingTurns float64) float64 {
	penalty := p.migrationPenalty(ep)
	return penalty + p.queueDelay(ep, penalty)*remainingTurns
}

// remainingTurns returns R for the request's session: how many turns it is
// expected to still take, counting the one being scheduled. Falls back to
// defaultRemainingTurns whenever the session is unknown or its turn index
// carries too few observations to divide by.
func (p *Plugin) remainingTurns(request *fwksched.InferenceRequest) float64 {
	if p.survival == nil || request == nil {
		return defaultRemainingTurns
	}
	state, ok := sessionstate.ReadSessionState(request)
	if !ok {
		return defaultRemainingTurns
	}
	if estimate, ok := p.survival.estimate(state.TurnsTaken); ok {
		return estimate
	}
	return defaultRemainingTurns
}

// PreRequest records that the session reached its current turn. This is the only
// hook that runs exactly once per dispatched request: Filter runs once per
// scheduling profile, so a plugin referenced by both the prefill and decode
// profiles would count a single request as several turns.
func (p *Plugin) PreRequest(_ context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) error {
	if p.survival == nil || request == nil {
		return nil
	}
	if state, ok := sessionstate.ReadSessionState(request); ok {
		p.survival.observe(state.TurnsTaken)
	}
	return nil
}

// migrationPenalty returns the prefill cost (ms) of the tokens an endpoint has
// not cached. Endpoints missing UncachedRequestTokens contribute no signal and
// are never the cheapest, matching endpointTTFT's handling of an absent
// prediction: treating an unknown as zero would make migration look free.
func (p *Plugin) migrationPenalty(ep fwksched.Endpoint) float64 {
	raw, ok := ep.Get(p.uncachedRequestTokensDataKey)
	if !ok {
		return math.MaxFloat64
	}
	uncached, ok := raw.(*attrconcurrency.UncachedRequestTokens)
	if !ok || uncached == nil {
		return math.MaxFloat64
	}
	return float64(uncached.Tokens) / p.config.PeakPrefillThroughput * 1000
}

// queueDelay returns an endpoint's wait (ms) excluding the prefill of the request
// being scheduled, which is charged once as the migration penalty rather than on
// every turn. The throughput estimate already excludes it; the latency
// predictor's TTFT is derived from this request's length and the endpoint's
// prefix cache score, so it contains the penalty and has it subtracted back out.
func (p *Plugin) queueDelay(ep fwksched.Endpoint, penalty float64) float64 {
	if !p.config.usesLatencyPredictor() {
		return p.endpointTTFT(ep)
	}
	return math.Max(p.endpointTTFT(ep)-penalty, 0)
}

// inFlightTokens returns an endpoint's in-flight token count, or 0 when the
// attribute is absent (no observed load).
func (p *Plugin) inFlightTokens(ep fwksched.Endpoint) int64 {
	if raw, ok := ep.Get(p.inFlightLoadDataKey); ok {
		if load, ok := raw.(*attrconcurrency.InFlightLoad); ok && load != nil {
			return load.Tokens
		}
	}
	return 0
}
