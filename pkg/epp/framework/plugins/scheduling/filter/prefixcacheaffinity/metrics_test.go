/*
Copyright 2026 The Kubernetes Authors.

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

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestRegisterMetrics(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	registry := prometheus.NewRegistry()
	require.NoError(t, registerMetrics(registry))
	require.NoError(t, registerMetrics(registry), "re-registering the same collector is tolerated")
	require.NoError(t, registerMetrics(nil), "a nil registerer is a no-op")
}

// Each Filter branch records exactly one decision outcome, so the counter labels
// distinguish an affinity-driven route from one where load reopened the set.
func TestFilterRecordsDecisionOutcome(t *testing.T) {
	tests := []struct {
		name    string
		config  Config
		input   []fwksched.Endpoint
		outcome string
	}{
		{
			name:   "single candidate is not applicable",
			config: Config{AffinityThreshold: 0.80},
			input:  []fwksched.Endpoint{makeEndpoint("a", 90, 10, 0, -1)},
			// A single candidate has no set to narrow.
			outcome: outcomeNotApplicable,
		},
		{
			name:    "disabled threshold is not applicable",
			config:  Config{AffinityThreshold: 0},
			input:   []fwksched.Endpoint{makeEndpoint("a", 90, 10, 0, -1), makeEndpoint("b", 10, 10, 0, -1)},
			outcome: outcomeNotApplicable,
		},
		{
			name:    "exploration skip",
			config:  Config{AffinityThreshold: 0.80, ExplorationProbability: 1.0},
			input:   []fwksched.Endpoint{makeEndpoint("a", 90, 100, 0, -1), makeEndpoint("b", 10, 50, 0, -1)},
			outcome: outcomeExploration,
		},
		{
			name:    "no endpoint meets the threshold",
			config:  Config{AffinityThreshold: 0.80},
			input:   []fwksched.Endpoint{makeEndpoint("a", 10, 10, 0, -1), makeEndpoint("b", 20, 20, 0, -1)},
			outcome: outcomeNoMatch,
		},
		{
			name:    "sticky set kept",
			config:  Config{AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 5000, TTFTSource: TTFTSourceLatencyPredictor},
			input:   []fwksched.Endpoint{makeEndpoint("a", 90, 100, 0, -1), makeEndpoint("b", 85, 120, 0, -1), makeEndpoint("c", 10, 50, 0, -1)},
			outcome: outcomeSticky,
		},
		{
			name:    "load gate reopens the set",
			config:  Config{AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 100, TTFTSource: TTFTSourceLatencyPredictor},
			input:   []fwksched.Endpoint{makeEndpoint("a", 90, 500, 0, -1), makeEndpoint("b", 10, 50, 0, -1)},
			outcome: outcomeLoadOverride,
		},
		{
			name:    "missing signal returns all endpoints",
			config:  Config{AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 100, TTFTSource: TTFTSourceLatencyPredictor},
			input:   []fwksched.Endpoint{makeEndpoint("a", 90, -1, 0, -1), makeEndpoint("b", 10, 50, 0, -1)},
			outcome: outcomeMissingSignal,
		},
	}

	allOutcomes := []string{outcomeSticky, outcomeNoMatch, outcomeLoadOverride, outcomeExploration, outcomeNotApplicable, outcomeMissingSignal}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resetMetrics()
			t.Cleanup(resetMetrics)

			p := newTestPlugin(test.config)
			p.Filter(context.Background(), nil, test.input)

			for _, outcome := range allOutcomes {
				want := float64(0)
				if outcome == test.outcome {
					want = 1
				}
				got := testutil.ToFloat64(filterDecisions.WithLabelValues("test", outcome))
				assert.Equalf(t, want, got, "outcome %q", outcome)
			}
		})
	}
}

// The served endpoint's estimate is recorded with the measured TTFT at the
// first response chunk; an endpoint without an estimate records nothing.
func TestResponseBodyRecordsTTFTEstimate(t *testing.T) {
	resetMetrics()
	t.Cleanup(resetMetrics)

	p := newTestPlugin(Config{AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 2000, TTFTSource: TTFTSourcePrefillThroughput,
		PeakPrefillThroughput: 1000, TTFTEstimate: TTFTEstimateFull})
	p.pluginState = fwkplugin.NewPluginState(t.Context())
	endpoints := []fwksched.Endpoint{makeEndpoint("a", 90, -1, 1500, 500), makeEndpoint("b", 10, -1, 0, 0)}

	request := &fwksched.InferenceRequest{RequestID: "served-a"}
	p.Filter(context.Background(), request, endpoints)
	served := endpoints[0].GetMetadata()
	p.ResponseBody(context.Background(), request, &requestcontrol.Response{StartOfStream: true}, served)

	require.Equal(t, 1, testutil.CollectAndCount(estimatedTTFT))
	assert.InDelta(t, 2.0, histogramSum(t, estimatedTTFT), 1e-9, "(1500 + 500) tokens / 1000 tokens/s")
	assert.Equal(t, uint64(1), histogramCount(t, actualTTFT))
	_, err := p.pluginState.Read(request.RequestID, ttftEstimateStateKey)
	assert.ErrorIs(t, err, fwkplugin.ErrNotFound, "the estimates are released at the first chunk")

	other := &fwksched.InferenceRequest{RequestID: "served-elsewhere"}
	p.Filter(context.Background(), other, endpoints)
	unknown := &fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: "c", Namespace: "default"}}
	p.ResponseBody(context.Background(), other, &requestcontrol.Response{StartOfStream: true}, unknown)
	assert.Equal(t, uint64(1), histogramCount(t, estimatedTTFT), "an endpoint without an estimate is not recorded")

	predictor := newTestPlugin(Config{AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 2000, TTFTSource: TTFTSourceLatencyPredictor})
	predictor.pluginState = fwkplugin.NewPluginState(t.Context())
	predicted := &fwksched.InferenceRequest{RequestID: "predictor"}
	predictor.Filter(context.Background(), predicted, []fwksched.Endpoint{makeEndpoint("a", 90, 100, 0, -1), makeEndpoint("b", 10, 50, 0, -1)})
	predictor.ResponseBody(context.Background(), predicted, &requestcontrol.Response{StartOfStream: true}, served)
	assert.Equal(t, uint64(1), histogramCount(t, estimatedTTFT), "the latency predictor source records nothing")
}

func histogramCount(t *testing.T, h *prometheus.HistogramVec) uint64 {
	t.Helper()
	m := &dto.Metric{}
	require.NoError(t, h.WithLabelValues("test").(prometheus.Histogram).Write(m))
	return m.GetHistogram().GetSampleCount()
}

func histogramSum(t *testing.T, h *prometheus.HistogramVec) float64 {
	t.Helper()
	m := &dto.Metric{}
	require.NoError(t, h.WithLabelValues("test").(prometheus.Histogram).Write(m))
	return m.GetHistogram().GetSampleSum()
}
