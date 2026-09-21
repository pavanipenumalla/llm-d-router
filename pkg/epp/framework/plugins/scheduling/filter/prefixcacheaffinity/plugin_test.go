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

package prefixcacheaffinity

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionstate"
)

// makeEndpoint creates a test endpoint with the given prefix cache match ratio
// (prefixMatch out of 100 total blocks), predicted TTFT, and in-flight tokens.
func makeEndpoint(name string, prefixMatch int, ttft float64, tokens int64) fwksched.Endpoint {
	meta := &fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Name: name, Namespace: "default"},
	}
	ep := fwksched.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	if prefixMatch >= 0 {
		ep.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(prefixMatch, 100, 16))
	}
	if ttft >= 0 {
		ep.Put(attrlatency.LatencyPredictionInfoDataKey, attrlatency.NewLatencyPredictionInfo(true, true, 0, 0, ttft, 0, 0))
	}
	if tokens >= 0 {
		ep.Put(attrconcurrency.InFlightLoadDataKey, &attrconcurrency.InFlightLoad{Tokens: tokens})
	}
	return ep
}

// withUncachedTokens attaches UncachedRequestTokens to an endpoint built by
// makeEndpoint, so session-cost tests can set the per-endpoint prefill cost
// without changing makeEndpoint's signature.
func withUncachedTokens(ep fwksched.Endpoint, tokens int64) fwksched.Endpoint {
	ep.Put(attrconcurrency.UncachedRequestTokensDataKey, &attrconcurrency.UncachedRequestTokens{Tokens: tokens})
	return ep
}

func newTestPlugin(config Config) *Plugin {
	p := &Plugin{
		typedName:                    fwkplugin.TypedName{Type: PluginType, Name: "test"},
		config:                       config,
		prefixMatchDataKey:           attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(config.PrefixMatchInfoProducerName),
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(config.LatencyPredictionInfoProducerName),
		inFlightLoadDataKey:          attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
		uncachedRequestTokensDataKey: attrconcurrency.UncachedRequestTokensDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
	}
	if config.usesSessionCost() {
		p.survival = newTurnSurvival()
		// Freeze the clock so seeded counts stay exact. Decay would otherwise
		// shave a fraction off them between the seed and the read, which is
		// enough to fail a sample-count check seeded at the threshold. Decay is
		// exercised directly in the estimator's own tests.
		frozen := p.survival.anchor
		p.survival.now = func() time.Time { return frozen }
	}
	return p
}

// requestAtTurn builds a request carrying the session state a session has after
// completing turn turns, which is what the session-state-producer publishes
// before dispatching turn+1.
func requestAtTurn(turn int64) *fwksched.InferenceRequest {
	request := &fwksched.InferenceRequest{}
	request.PutAttribute(sessionstate.SessionStateDataKey, sessionstate.SessionState{TurnsTaken: turn})
	return request
}

// seedRemainingTurns fills the survival table so estimate(turn) returns exactly
// r. Setting counts[turn..turn+r-1] to one value m leaves the tail sum at m*r
// over a denominator of m, and m of minTurnSamples clears the sample-count gate.
func seedRemainingTurns(p *Plugin, turn int64, r int64) {
	for i := turn; i < turn+r; i++ {
		p.survival.counts[turnIndex(i)] = minTurnSamples
	}
}

func TestFilter_AffinityThresholdDisabled(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 0, 10, 0),
		makeEndpoint("b", 90, 20, 0),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 2, len(result), "affinityThreshold=0 should return all")
}

func TestFilter_SingleEndpoint(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80})
	endpoints := []fwksched.Endpoint{makeEndpoint("a", 90, 10, 0)}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 1, len(result), "single endpoint should always pass")
}

func TestFilter_NoStickyEndpoints(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 0})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 10, 10, 0),
		makeEndpoint("b", 20, 20, 0),
		makeEndpoint("c", 50, 30, 0),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 3, len(result), "no sticky endpoints should return all")
}

func TestFilter_NarrowToSticky(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 0, MaxTTFTPenaltyMs: 5000, TTFTSource: TTFTSourceLatencyPredictor})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 90, 100, 0),
		makeEndpoint("b", 85, 120, 0),
		makeEndpoint("c", 10, 50, 0),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 2, len(result), "should narrow to sticky endpoints")
}

func TestFilter_TTFTPenaltyBreaksStickiness(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 0, MaxTTFTPenaltyMs: 100, TTFTSource: TTFTSourceLatencyPredictor})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 90, 500, 0),
		makeEndpoint("b", 10, 50, 0),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 2, len(result), "TTFT penalty should break stickiness")
}

// With PeakPrefillThroughput=1000 tokens/sec, in-flight tokens map to TTFT as
// tokens/1000*1000 = tokens ms: endpoint "a" -> 500ms, "b" -> 50ms.
func TestFilter_ThroughputTTFTBreaksStickiness(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 0, MaxTTFTPenaltyMs: 100, TTFTSource: TTFTSourcePrefillThroughput, PeakPrefillThroughput: 1000})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 90, 10, 500),
		makeEndpoint("b", 10, 10, 50),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 2, len(result), "throughput-derived TTFT penalty should break stickiness")
}

func TestFilter_ThroughputTTFTWithinThreshold(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 0, MaxTTFTPenaltyMs: 1000, TTFTSource: TTFTSourcePrefillThroughput, PeakPrefillThroughput: 1000})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 90, 10, 500),
		makeEndpoint("b", 10, 10, 50),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 1, len(result), "throughput-derived TTFT within threshold should NOT break stickiness")
	assert.Equal(t, "a", result[0].GetMetadata().ID.Name)
}

func TestFilter_TTFTPenaltyDisabled(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 0, MaxTTFTPenaltyMs: 0, TTFTSource: TTFTSourcePrefillThroughput, PeakPrefillThroughput: 1000})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 90, 10, 5000), // Huge load
		makeEndpoint("b", 10, 10, 50),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 1, len(result), "maxTTFTPenaltyMs=0 should NOT break stickiness")
	assert.Equal(t, "a", result[0].GetMetadata().ID.Name)
}

func TestFilter_ExplorationProbability(t *testing.T) {
	p := newTestPlugin(Config{AffinityThreshold: 0.80, ExplorationProbability: 1.0})
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 90, 100, 0),
		makeEndpoint("b", 10, 50, 0),
	}
	result := p.Filter(context.Background(), nil, endpoints)
	assert.Equal(t, 2, len(result), "epsilon=1.0 should always skip gate")
}

func TestConsumes_ConditionalAttributes(t *testing.T) {
	// Gate disabled: neither TTFT source is consumed.
	p := newTestPlugin(Config{MaxTTFTPenaltyMs: 0})
	consumed := p.Consumes()
	_, ok := consumed.Required[p.inFlightLoadDataKey]
	assert.False(t, ok, "InFlightLoadDataKey should not be consumed when the gate is disabled")
	_, ok = consumed.Required[p.latencyPredictionInfoDataKey]
	assert.False(t, ok, "LatencyPredictionInfoDataKey should not be consumed when the gate is disabled")

	// Gate using the latency predictor.
	p = newTestPlugin(Config{MaxTTFTPenaltyMs: 5000, TTFTSource: TTFTSourceLatencyPredictor})
	consumed = p.Consumes()
	_, ok = consumed.Required[p.latencyPredictionInfoDataKey]
	assert.True(t, ok)
	_, ok = consumed.Required[p.inFlightLoadDataKey]
	assert.False(t, ok)

	// Gate using peak prefill throughput.
	p = newTestPlugin(Config{MaxTTFTPenaltyMs: 5000, TTFTSource: TTFTSourcePrefillThroughput, PeakPrefillThroughput: 1000})
	consumed = p.Consumes()
	_, ok = consumed.Required[p.inFlightLoadDataKey]
	assert.True(t, ok)
	_, ok = consumed.Required[p.latencyPredictionInfoDataKey]
	assert.False(t, ok)
}

func TestFactory_ValidConfig(t *testing.T) {
	plugin, err := Factory("test", fwkplugin.StrictDecoder(nil), nil)
	assert.NoError(t, err)
	assert.NotNil(t, plugin)
	assert.Equal(t, PluginType, plugin.TypedName().Type)
}

func TestFactory_PartialConfigPreservesDefaults(t *testing.T) {
	// Setting only affinityThreshold should preserve defaults for other params.
	plugin, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"affinityThreshold": 0.95}`)), nil)
	assert.NoError(t, err)
	p := plugin.(*Plugin)
	assert.Equal(t, 0.95, p.config.AffinityThreshold)
	assert.Equal(t, DefaultConfig.ExplorationProbability, p.config.ExplorationProbability)
	assert.Equal(t, DefaultConfig.MaxTTFTPenaltyMs, p.config.MaxTTFTPenaltyMs)

	// Setting only explorationProbability should preserve defaults for other params.
	plugin, err = Factory("test", fwkplugin.StrictDecoder([]byte(`{"explorationProbability": 0.05}`)), nil)
	assert.NoError(t, err)
	p = plugin.(*Plugin)
	assert.Equal(t, DefaultConfig.AffinityThreshold, p.config.AffinityThreshold)
	assert.Equal(t, 0.05, p.config.ExplorationProbability)
	assert.Equal(t, DefaultConfig.MaxTTFTPenaltyMs, p.config.MaxTTFTPenaltyMs)

	// Setting only maxTTFTPenaltyMs should preserve defaults for other params.
	plugin, err = Factory("test", fwkplugin.StrictDecoder([]byte(`{"maxTTFTPenaltyMs": 10000}`)), nil)
	assert.NoError(t, err)
	p = plugin.(*Plugin)
	assert.Equal(t, DefaultConfig.AffinityThreshold, p.config.AffinityThreshold)
	assert.Equal(t, DefaultConfig.ExplorationProbability, p.config.ExplorationProbability)
	assert.Equal(t, float64(10000), p.config.MaxTTFTPenaltyMs)
	assert.Equal(t, DefaultConfig.TTFTSource, p.config.TTFTSource)
	assert.Equal(t, DefaultConfig.PeakPrefillThroughput, p.config.PeakPrefillThroughput)
}

func TestFactory_InvalidAffinityThreshold(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  string
	}{
		{
			name: "below zero",
			raw:  `{"affinityThreshold": -0.1}`,
		},
		{
			name: "above one",
			raw:  `{"affinityThreshold": 1.5}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Factory("test", fwkplugin.StrictDecoder([]byte(tc.raw)), nil)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "affinityThreshold must be in [0, 1]")
		})
	}
}

func TestFactory_InvalidExplorationProbability(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"explorationProbability": -0.1}`)), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "explorationProbability must be in [0, 1]")
}

func TestFactory_InvalidPeakPrefillThroughput(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"peakPrefillThroughput": -1}`)), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "peakPrefillThroughput must be >= 0")
}

// The throughput TTFT source needs a non-zero divisor: with the gate enabled
// (maxTTFTPenaltyMs defaults to 5000) and ttftSource=prefillThroughput,
// peakPrefillThroughput=0 must be rejected.
func TestFactory_ThroughputModeRequiresPeakPrefillThroughput(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"ttftSource": "prefillThroughput", "peakPrefillThroughput": 0}`)), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "peakPrefillThroughput must be > 0 when ttftSource is prefillThroughput")
}

func TestFactory_ThroughputModeValid(t *testing.T) {
	plugin, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"ttftSource": "prefillThroughput", "peakPrefillThroughput": 1000}`)), nil)
	assert.NoError(t, err)
	p := plugin.(*Plugin)
	assert.Equal(t, TTFTSourcePrefillThroughput, p.config.TTFTSource)
	assert.Equal(t, float64(1000), p.config.PeakPrefillThroughput)
}

// An unrecognized ttftSource value is rejected rather than silently treated as
// the default.
func TestFactory_InvalidTTFTSource(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"ttftSource": "bogus"}`)), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "ttftSource must be")
}

// An empty ttftSource is rejected rather than silently defaulted: the default is
// supplied by DefaultConfig, so an explicit empty value is a configuration error.
func TestFactory_EmptyTTFTSourceRejected(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"ttftSource": ""}`)), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "ttftSource must be")
}

// peakPrefillThroughput=0 is valid as long as the throughput source is unused:
// either the gate is disabled (maxTTFTPenaltyMs=0) or the latency predictor
// supplies TTFT.
func TestFactory_ZeroPeakPrefillThroughputAllowedWhenUnused(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"maxTTFTPenaltyMs": 0, "ttftSource": "prefillThroughput", "peakPrefillThroughput": 0}`)), nil)
	assert.NoError(t, err, "throughput source unused when the gate is disabled")

	_, err = Factory("test", fwkplugin.StrictDecoder([]byte(`{"ttftSource": "latencyPredictor", "peakPrefillThroughput": 0}`)), nil)
	assert.NoError(t, err, "throughput source unused when the latency predictor supplies TTFT")
}

// The default TTFT source is prefillThroughput, so an unset ttftSource selects
// the throughput estimate: it consumes InFlightLoad and requires a non-zero
// peakPrefillThroughput when the gate is enabled.
func TestFactory_DefaultsToPrefillThroughput(t *testing.T) {
	assert.Equal(t, TTFTSourcePrefillThroughput, DefaultConfig.TTFTSource)

	plugin, err := Factory("test", fwkplugin.StrictDecoder(nil), nil)
	assert.NoError(t, err)
	p := plugin.(*Plugin)
	assert.Equal(t, TTFTSourcePrefillThroughput, p.config.TTFTSource)

	_, err = Factory("test", fwkplugin.StrictDecoder([]byte(`{"peakPrefillThroughput": 0}`)), nil)
	assert.Error(t, err, "default throughput source needs a non-zero peakPrefillThroughput")
	assert.Contains(t, err.Error(), "peakPrefillThroughput must be > 0")
}

// Session cost tests all use PeakPrefillThroughput=1000, so a token count maps
// to milliseconds one to one: tokens/1000*1000 = tokens ms. Endpoint "a" is
// sticky (90 of 100 blocks matched), "b" is not (10 of 100). R is seeded into
// the survival table instead of configured, and the request carries the turn
// index the estimate is read at.

// L(a) = 100 + 2000*15 = 30100, L(b) = 45000 + 0*15 = 45000. Rebuilding 45s of
// history costs more than 15 turns behind a 2s queue, so the session stays.
func TestFilter_SessionCostStaysOnStickyEndpoint(t *testing.T) {
	p := newTestPlugin(Config{
		AffinityThreshold: 0.80, TTFTSource: TTFTSourcePrefillThroughput,
		PeakPrefillThroughput: 1000, PenaltySource: PenaltySourceSessionCost,
	})
	seedRemainingTurns(p, 0, 15)
	endpoints := []fwksched.Endpoint{
		withUncachedTokens(makeEndpoint("a", 90, -1, 2000), 100),
		withUncachedTokens(makeEndpoint("b", 10, -1, 0), 45000),
	}
	result := p.Filter(context.Background(), requestAtTurn(0), endpoints)
	assert.Equal(t, 1, len(result))
	assert.Equal(t, "a", result[0].GetMetadata().ID.Name)
}

// L(a) = 100 + 4000*15 = 60100, L(b) = 45000. Past a 3s queue on the sticky
// endpoint the one-time rebuild is cheaper, so the filter returns the cold set.
func TestFilter_SessionCostMigratesWhenStickySaturated(t *testing.T) {
	p := newTestPlugin(Config{
		AffinityThreshold: 0.80, TTFTSource: TTFTSourcePrefillThroughput,
		PeakPrefillThroughput: 1000, PenaltySource: PenaltySourceSessionCost,
	})
	seedRemainingTurns(p, 0, 15)
	endpoints := []fwksched.Endpoint{
		withUncachedTokens(makeEndpoint("a", 90, -1, 4000), 100),
		withUncachedTokens(makeEndpoint("b", 10, -1, 0), 45000),
	}
	result := p.Filter(context.Background(), requestAtTurn(0), endpoints)
	assert.Equal(t, 1, len(result))
	assert.Equal(t, "b", result[0].GetMetadata().ID.Name)
}

// Same load as the stay case, but a short history: L(a) = 30100, L(b) = 1000.
// A young session moves under load a long one would sit through.
func TestFilter_SessionCostMigratesShortHistory(t *testing.T) {
	p := newTestPlugin(Config{
		AffinityThreshold: 0.80, TTFTSource: TTFTSourcePrefillThroughput,
		PeakPrefillThroughput: 1000, PenaltySource: PenaltySourceSessionCost,
	})
	seedRemainingTurns(p, 0, 15)
	endpoints := []fwksched.Endpoint{
		withUncachedTokens(makeEndpoint("a", 90, -1, 2000), 100),
		withUncachedTokens(makeEndpoint("b", 10, -1, 0), 1000),
	}
	result := p.Filter(context.Background(), requestAtTurn(0), endpoints)
	assert.Equal(t, 1, len(result))
	assert.Equal(t, "b", result[0].GetMetadata().ID.Name)
}

// At one remaining turn the cost reduces to uncached + inflight, which is the
// quantity token-load-scorer ranks by. Both directions are checked.
func TestFilter_SessionCostOneRemainingTurnIsTokenSum(t *testing.T) {
	config := Config{
		AffinityThreshold: 0.80, TTFTSource: TTFTSourcePrefillThroughput,
		PeakPrefillThroughput: 1000, PenaltySource: PenaltySourceSessionCost,
	}

	// L(a) = 100+2000 = 2100, L(b) = 1000+0 = 1000.
	p := newTestPlugin(config)
	seedRemainingTurns(p, 0, 1)
	result := p.Filter(context.Background(), requestAtTurn(0), []fwksched.Endpoint{
		withUncachedTokens(makeEndpoint("a", 90, -1, 2000), 100),
		withUncachedTokens(makeEndpoint("b", 10, -1, 0), 1000),
	})
	assert.Equal(t, 1, len(result))
	assert.Equal(t, "b", result[0].GetMetadata().ID.Name)

	// L(a) = 100+100 = 200, L(b) = 1000+0 = 1000.
	p = newTestPlugin(config)
	seedRemainingTurns(p, 0, 1)
	result = p.Filter(context.Background(), requestAtTurn(0), []fwksched.Endpoint{
		withUncachedTokens(makeEndpoint("a", 90, -1, 100), 100),
		withUncachedTokens(makeEndpoint("b", 10, -1, 0), 1000),
	})
	assert.Equal(t, 1, len(result))
	assert.Equal(t, "a", result[0].GetMetadata().ID.Name)
}

// The predictor's TTFT already contains the endpoint's prefill cost, so queue is
// TTFT - M: a -> 2100-100 = 2000, b -> 45000-45000 = 0. That reproduces the
// throughput case exactly, L(a) = 30100 against L(b) = 45000.
func TestFilter_SessionCostLatencyPredictorSubtractsPenalty(t *testing.T) {
	p := newTestPlugin(Config{
		AffinityThreshold: 0.80, TTFTSource: TTFTSourceLatencyPredictor,
		PeakPrefillThroughput: 1000, PenaltySource: PenaltySourceSessionCost,
	})
	seedRemainingTurns(p, 0, 15)
	endpoints := []fwksched.Endpoint{
		withUncachedTokens(makeEndpoint("a", 90, 2100, -1), 100),
		withUncachedTokens(makeEndpoint("b", 10, 45000, -1), 45000),
	}
	result := p.Filter(context.Background(), requestAtTurn(0), endpoints)
	assert.Equal(t, 1, len(result))
	assert.Equal(t, "a", result[0].GetMetadata().ID.Name)
}

// Endpoint "a" has M=3000 against a predicted TTFT of 50, so the subtraction
// goes negative and must clamp to zero: L(a) = 3000, L(b) = 2500, and the
// session moves. Unclamped, L(a) would be 3000 + (-2950*15) = -41250 and win.
func TestFilter_SessionCostClampsNegativeQueueDelay(t *testing.T) {
	p := newTestPlugin(Config{
		AffinityThreshold: 0.80, TTFTSource: TTFTSourceLatencyPredictor,
		PeakPrefillThroughput: 1000, PenaltySource: PenaltySourceSessionCost,
	})
	seedRemainingTurns(p, 0, 15)
	endpoints := []fwksched.Endpoint{
		withUncachedTokens(makeEndpoint("a", 90, 50, -1), 3000),
		withUncachedTokens(makeEndpoint("b", 10, 2500, -1), 2500),
	}
	result := p.Filter(context.Background(), requestAtTurn(0), endpoints)
	assert.Equal(t, 1, len(result))
	assert.Equal(t, "b", result[0].GetMetadata().ID.Name)
}

// maxTTFTPenaltyMs=0 means "never break stickiness" in constant mode. Under
// sessionCost it carries no meaning, so the migration still happens.
func TestFilter_SessionCostIgnoresMaxTTFTPenalty(t *testing.T) {
	p := newTestPlugin(Config{
		AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 0,
		TTFTSource: TTFTSourcePrefillThroughput, PeakPrefillThroughput: 1000,
		PenaltySource: PenaltySourceSessionCost,
	})
	seedRemainingTurns(p, 0, 15)
	endpoints := []fwksched.Endpoint{
		withUncachedTokens(makeEndpoint("a", 90, -1, 4000), 100),
		withUncachedTokens(makeEndpoint("b", 10, -1, 0), 45000),
	}
	result := p.Filter(context.Background(), requestAtTurn(0), endpoints)
	assert.Equal(t, 1, len(result))
	assert.Equal(t, "b", result[0].GetMetadata().ID.Name)
}

// An endpoint without UncachedRequestTokens carries no prefill signal and is
// never the cheapest. With the attribute absent everywhere the comparison
// cannot favour the cold set, so the session stays where its cache is.
func TestFilter_SessionCostMissingUncachedTokensStaysSticky(t *testing.T) {
	p := newTestPlugin(Config{
		AffinityThreshold: 0.80, TTFTSource: TTFTSourcePrefillThroughput,
		PeakPrefillThroughput: 1000, PenaltySource: PenaltySourceSessionCost,
	})
	seedRemainingTurns(p, 0, 15)
	endpoints := []fwksched.Endpoint{
		makeEndpoint("a", 90, -1, 4000),
		makeEndpoint("b", 10, -1, 0),
	}
	result := p.Filter(context.Background(), requestAtTurn(0), endpoints)
	assert.Equal(t, 1, len(result))
	assert.Equal(t, "a", result[0].GetMetadata().ID.Name)
}

// A request with no session state, and a session at a turn index with too few
// observations, both fall back to defaultRemainingTurns rather than guessing.
//
// R multiplies the per-turn queue and nothing else, so with the sticky endpoint
// the warmer but busier one, a larger R argues for moving: L(a) = 100 + 2000R
// against a flat L(b) = 5000 crosses over at R = 2.45. The estimate and the
// fallback therefore land on opposite sides, which is what makes the fallback
// observable here rather than silent.
func TestFilter_SessionCostFallsBackWithoutUsableTurnData(t *testing.T) {
	config := Config{
		AffinityThreshold: 0.80, TTFTSource: TTFTSourcePrefillThroughput,
		PeakPrefillThroughput: 1000, PenaltySource: PenaltySourceSessionCost,
	}
	endpoints := func() []fwksched.Endpoint {
		return []fwksched.Endpoint{
			withUncachedTokens(makeEndpoint("a", 90, -1, 2000), 100),
			withUncachedTokens(makeEndpoint("b", 10, -1, 0), 5000),
		}
	}

	// Estimate available: R = 15, so 30100 against 5000 and the session moves.
	// Establishes that these figures discriminate before the fallback is tested.
	p := newTestPlugin(config)
	seedRemainingTurns(p, 0, 15)
	result := p.Filter(context.Background(), requestAtTurn(0), endpoints())
	assert.Equal(t, "b", result[0].GetMetadata().ID.Name)

	// No session state on the request: R = 1, so 2100 against 5000 and it stays.
	p = newTestPlugin(config)
	seedRemainingTurns(p, 0, 15)
	result = p.Filter(context.Background(), &fwksched.InferenceRequest{}, endpoints())
	assert.Equal(t, "a", result[0].GetMetadata().ID.Name, "absent session state must fall back to R=1")

	// Session state present, but turn 20 lies past every seeded index, so that
	// turn count carries no observations to divide by.
	p = newTestPlugin(config)
	seedRemainingTurns(p, 0, 15)
	result = p.Filter(context.Background(), requestAtTurn(20), endpoints())
	assert.Equal(t, "a", result[0].GetMetadata().ID.Name, "an unobserved turn index must fall back to R=1")
}

// PreRequest is the only per-request hook, and it must stay inert in constant
// mode because the plugin is registered for it regardless of configuration.
func TestPreRequest_RecordsTurnOnlyUnderSessionCost(t *testing.T) {
	p := newTestPlugin(Config{
		AffinityThreshold: 0.80, TTFTSource: TTFTSourcePrefillThroughput,
		PeakPrefillThroughput: 1000, PenaltySource: PenaltySourceSessionCost,
	})
	for i := 0; i < minTurnSamples; i++ {
		assert.NoError(t, p.PreRequest(context.Background(), requestAtTurn(3), nil))
	}
	assert.Equal(t, float64(minTurnSamples), p.survival.counts[3])

	// A request without session state contributes nothing.
	assert.NoError(t, p.PreRequest(context.Background(), &fwksched.InferenceRequest{}, nil))
	assert.Equal(t, float64(minTurnSamples), p.survival.counts[3])

	constant := newTestPlugin(Config{AffinityThreshold: 0.80, PenaltySource: PenaltySourceConstant})
	assert.Nil(t, constant.survival, "constant mode allocates no table")
	assert.NoError(t, constant.PreRequest(context.Background(), requestAtTurn(3), nil))
}

// sessionCost reads a TTFT source, UncachedRequestTokens and SessionState
// regardless of maxTTFTPenaltyMs, so all must be declared even with the constant
// gate off.
func TestConsumes_SessionCostAttributes(t *testing.T) {
	p := newTestPlugin(Config{
		MaxTTFTPenaltyMs: 0, TTFTSource: TTFTSourcePrefillThroughput,
		PeakPrefillThroughput: 1000, PenaltySource: PenaltySourceSessionCost,
	})
	consumed := p.Consumes()
	_, ok := consumed.Required[p.uncachedRequestTokensDataKey]
	assert.True(t, ok, "UncachedRequestTokensDataKey should be consumed under sessionCost")
	_, ok = consumed.Required[p.inFlightLoadDataKey]
	assert.True(t, ok, "the throughput source is still read under sessionCost")
	_, ok = consumed.Required[sessionstate.SessionStateDataKey]
	assert.True(t, ok, "the turn count comes from SessionState")

	p = newTestPlugin(Config{
		MaxTTFTPenaltyMs: 0, TTFTSource: TTFTSourceLatencyPredictor,
		PeakPrefillThroughput: 1000, PenaltySource: PenaltySourceSessionCost,
	})
	consumed = p.Consumes()
	_, ok = consumed.Required[p.uncachedRequestTokensDataKey]
	assert.True(t, ok)
	_, ok = consumed.Required[p.latencyPredictionInfoDataKey]
	assert.True(t, ok)

	// Constant mode acquires none of them.
	p = newTestPlugin(Config{MaxTTFTPenaltyMs: 0, PenaltySource: PenaltySourceConstant})
	_, ok = p.Consumes().Required[p.uncachedRequestTokensDataKey]
	assert.False(t, ok, "constant mode must not depend on the in-flight load producer")
	_, ok = p.Consumes().Required[sessionstate.SessionStateDataKey]
	assert.False(t, ok, "constant mode must not depend on the session-state producer")
}

func TestFactory_DefaultsToConstantPenaltySource(t *testing.T) {
	assert.Equal(t, PenaltySourceConstant, DefaultConfig.PenaltySource)

	plugin, err := Factory("test", fwkplugin.StrictDecoder(nil), nil)
	assert.NoError(t, err)
	assert.Equal(t, PenaltySourceConstant, plugin.(*Plugin).config.PenaltySource)
	assert.Nil(t, plugin.(*Plugin).survival)

	plugin, err = Factory("test", fwkplugin.StrictDecoder([]byte(
		`{"penaltySource": "sessionCost", "peakPrefillThroughput": 1000}`)), nil)
	assert.NoError(t, err)
	assert.NotNil(t, plugin.(*Plugin).survival)
}

func TestFactory_InvalidPenaltySource(t *testing.T) {
	for _, tc := range []struct {
		name string
		raw  string
	}{
		{
			name: "unknown value",
			raw:  `{"penaltySource": "nonsense"}`,
		},
		{
			name: "empty value",
			raw:  `{"penaltySource": ""}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Factory("test", fwkplugin.StrictDecoder([]byte(tc.raw)), nil)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "penaltySource must be")
		})
	}
}

// The prefill cost is converted from tokens to milliseconds on both TTFT paths,
// so sessionCost needs a non-zero throughput even with the latency predictor,
// where the constant gate allows zero.
func TestFactory_SessionCostRequiresPeakPrefillThroughput(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder([]byte(
		`{"penaltySource": "sessionCost", "ttftSource": "latencyPredictor", "peakPrefillThroughput": 0}`)), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "peakPrefillThroughput must be > 0")

	_, err = Factory("test", fwkplugin.StrictDecoder([]byte(
		`{"penaltySource": "sessionCost", "ttftSource": "latencyPredictor", "peakPrefillThroughput": 1000}`)), nil)
	assert.NoError(t, err)
}
