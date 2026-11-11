/*
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

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

func TestQueueWaitCorrection_Factor(t *testing.T) {
	now := time.Now()
	for _, tc := range []struct {
		name    string
		samples [][2]time.Duration // measured, estimated
		want    float64
	}{
		{name: "no samples", want: 1},
		{name: "too few samples", samples: [][2]time.Duration{{time.Second, 10 * time.Second}, {time.Second, 10 * time.Second}}, want: 1},
		{name: "estimate too high", samples: [][2]time.Duration{{time.Second, 2 * time.Second}, {time.Second, 2 * time.Second}, {2 * time.Second, 4 * time.Second}}, want: 0.5},
		{name: "lower bound", samples: [][2]time.Duration{{0, 10 * time.Second}, {0, 10 * time.Second}, {-time.Second, 10 * time.Second}}, want: minQueueWaitFactor},
		{name: "estimate too low is not raised", samples: [][2]time.Duration{{4 * time.Second, time.Second}, {4 * time.Second, time.Second}, {4 * time.Second, time.Second}}, want: 1},
		{name: "requests without a queue wait estimate are left out",
			samples: [][2]time.Duration{{time.Second, 2 * time.Second}, {time.Second, 2 * time.Second}, {5 * time.Second, 0}, {time.Second, 2 * time.Second}}, want: 0.5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &queueWaitCorrection{}
			for _, s := range tc.samples {
				c.add(now, s[0], s[1])
			}
			assert.InDelta(t, tc.want, c.factor(now), 1e-9)
		})
	}
}

func TestQueueWaitCorrection_Window(t *testing.T) {
	start := time.Now()
	c := &queueWaitCorrection{}
	for range 3 {
		c.add(start, time.Second, 4*time.Second)
	}
	assert.InDelta(t, 0.25, c.factor(start.Add(queueWaitCorrectionWindow)), 1e-9)
	assert.InDelta(t, 1, c.factor(start.Add(queueWaitCorrectionWindow+time.Second)), 1e-9, "samples older than the window are dropped")
	assert.Empty(t, c.samples)
}

// Requests whose measured queue wait is half the estimate halve the queue wait
// part of later estimates on every endpoint; the prefill part is unchanged.
func TestFilter_QueueWaitCorrectionLowersEstimate(t *testing.T) {
	config := Config{AffinityThreshold: 0.80, MaxTTFTPenaltyMs: 2000, TTFTSource: TTFTSourcePrefillThroughput,
		PeakPrefillThroughput: 1000, TTFTEstimate: TTFTEstimateFull, MaxNumSeqs: 8, QueueWaitCorrection: true}
	p := newTestPlugin(config)
	p.pluginState = fwkplugin.NewPluginState(t.Context())
	p.queueWait = &queueWaitCorrection{}
	// 10 in flight, 2 queued at 1 completion/s: 2000 ms queue wait; 4 pending
	// requests' 1000 tokens keep a (4 - 2) / 4 share: 500 ms prefill.
	load := attrconcurrency.InFlightLoad{Requests: 10, PendingRequests: 4, Tokens: 1000, CompletionsPerSecond: 1}
	ep := makeQueuedEndpoint("a", 90, load, 0, 0)
	ttft, ok := p.endpointTTFT(ep)
	assert.True(t, ok)
	assert.InDelta(t, 2500, ttft, 1e-6)

	for _, id := range []string{"r1", "r2", "r3"} {
		request := &fwksched.InferenceRequest{RequestID: id}
		p.Filter(context.Background(), request, []fwksched.Endpoint{ep})
		state, err := fwkplugin.ReadPluginStateKey[*ttftEstimates](p.pluginState, id, ttftEstimateStateKey)
		assert.NoError(t, err)
		// Measured TTFT of 1500 ms: 500 ms prefill + 1000 ms queue wait.
		state.at = time.Now().Add(-1500 * time.Millisecond)
		p.ResponseBody(context.Background(), request, &requestcontrol.Response{StartOfStream: true}, ep.GetMetadata())
	}
	ttft, ok = p.endpointTTFT(ep)
	assert.True(t, ok)
	assert.InDelta(t, 500+0.5*2000, ttft, 20, "queue wait halved, prefill unchanged")

	other := makeQueuedEndpoint("b", 10, attrconcurrency.InFlightLoad{Requests: 12, PendingRequests: 4, Tokens: 0, CompletionsPerSecond: 1}, 0, 0)
	ttft, ok = p.endpointTTFT(other)
	assert.True(t, ok)
	assert.InDelta(t, 0.5*4000, ttft, 20, "the correction applies to every endpoint")
}

func TestFactory_QueueWaitCorrection(t *testing.T) {
	plugin, err := Factory("test", fwkplugin.StrictDecoder([]byte(`{"ttftEstimate": "full", "maxNumSeqs": 8, "queueWaitCorrection": true}`)), nil)
	assert.NoError(t, err)
	p := plugin.(*Plugin)
	assert.True(t, p.config.QueueWaitCorrection)
	assert.Nil(t, p.queueWait, "no correction without plugin state (nil handle)")
}
