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
	"sync"
	"time"
)

const (
	// queueWaitCorrectionWindow is how long a measured queue wait counts toward the correction.
	queueWaitCorrectionWindow = 30 * time.Second
	// minQueueWaitSamples is the number of measured queue waits below which no correction applies.
	minQueueWaitSamples = 3
	// minQueueWaitFactor bounds how far the correction lowers the queue wait.
	minQueueWaitFactor = 0.25
)

type queueWaitSample struct {
	at        time.Time
	measured  time.Duration
	estimated time.Duration
}

// queueWaitCorrection scales the estimated queue wait by the ratio of measured
// to estimated queue wait of the requests that received their first token in
// the last queueWaitCorrectionWindow, across all endpoints. The completion
// rate behind the estimate lags when requests start finishing faster, which
// overestimates the wait on every endpoint at once; the ratio follows that
// within one wait. The factor is at most 1: a few requests that waited longer
// than estimated for other reasons do not raise it.
type queueWaitCorrection struct {
	mu      sync.Mutex
	samples []queueWaitSample
}

// add records a request's estimated queue wait and the part of its measured
// TTFT beyond the estimated prefill time. Requests without a queue wait estimate
// are not recorded.
func (c *queueWaitCorrection) add(now time.Time, measured, estimated time.Duration) {
	if estimated <= 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.prune(now)
	c.samples = append(c.samples, queueWaitSample{at: now, measured: max(measured, 0), estimated: estimated})
}

// factor returns the multiplier for estimated queue waits, in [minQueueWaitFactor, 1].
func (c *queueWaitCorrection) factor(now time.Time) float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.prune(now)
	if len(c.samples) < minQueueWaitSamples {
		return 1
	}
	var measured, estimated time.Duration
	for _, s := range c.samples {
		measured += s.measured
		estimated += s.estimated
	}
	return min(max(float64(measured)/float64(estimated), minQueueWaitFactor), 1)
}

func (c *queueWaitCorrection) prune(now time.Time) {
	i := 0
	for i < len(c.samples) && now.Sub(c.samples[i].at) > queueWaitCorrectionWindow {
		i++
	}
	c.samples = c.samples[i:]
}
