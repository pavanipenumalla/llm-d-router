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
	"math"
	"sync"
	"time"
)

// Tuning for the remaining-turns estimate used by PenaltySourceSessionCost.
// These are properties of the estimator rather than operator policy, so they are
// constants and not configuration.
const (
	// maxTrackedTurns bounds the survival table. Sessions deeper than this are
	// estimated at the last index, where the estimate has saturated to 1; by
	// that depth the prefill a session has accumulated dominates the comparison
	// on its own. 8KB per plugin instance.
	maxTrackedTurns = 1024

	// minTurnSamples is the observation count a turn index needs before its
	// estimate is trusted. The relative error of the estimate falls off as
	// 1/sqrt(n), so below this the ratio reflects one or two sessions'
	// idiosyncrasy rather than the population.
	minTurnSamples = 4

	// turnHistoryHalfLife is how long an observation takes to lose half its
	// weight. Tied to the session-state-producer's own eviction TTL: a session
	// keeps contributing observations until it stops arriving, so a shorter
	// half-life would fade its early turns before its later turns arrive and
	// bias the distribution short.
	turnHistoryHalfLife = time.Hour

	// defaultRemainingTurns applies until a turn index reaches minTurnSamples.
	// At 1 the cost reduces to M + queue, which asserts nothing about a
	// session's future rather than inventing a prior for it.
	defaultRemainingTurns = 1
)

// turnSurvival estimates how many turns a session has left, from the turn counts
// observed across every session an instance has seen.
//
// A session running L turns is observed once at each turn index 0 to L-1, so
// counts[t] is the number of sessions that reached turn t: the survival function
// of session length. Expected remaining turns at t, counting the request being
// scheduled, is its mean residual life:
//
//	R(t) = sum(counts[k] for k >= t) / counts[t]
//
// which is >= 1 without a floor, because counts[t] appears in its own numerator.
// Averaging the observed turn indices instead answers a different question, "how
// far along is a typical request", and converges below the true mean length
// because every reading is mid-session and long sessions contribute more
// readings.
type turnSurvival struct {
	mu       sync.Mutex
	counts   [maxTrackedTurns]float64
	anchor   time.Time
	halfLife time.Duration
	now      func() time.Time
}

func newTurnSurvival() *turnSurvival {
	return &turnSurvival{
		anchor:   time.Now(),
		halfLife: turnHistoryHalfLife,
		now:      time.Now,
	}
}

// turnIndex maps a turn count onto the table, saturating at the last slot.
func turnIndex(turn int64) int {
	if turn >= maxTrackedTurns {
		return maxTrackedTurns - 1
	}
	return int(turn)
}

// observe records that a session reached the given turn.
func (s *turnSurvival) observe(turn int64) {
	if turn < 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.decayLocked()
	s.counts[turnIndex(turn)]++
}

// estimate returns the expected remaining turns at the given turn count, and
// false when that index holds fewer than minTurnSamples observations.
func (s *turnSurvival) estimate(turn int64) (float64, bool) {
	if turn < 0 {
		return 0, false
	}
	idx := turnIndex(turn)

	s.mu.Lock()
	defer s.mu.Unlock()

	// The ratio below is invariant under a decay applied to every entry, so the
	// read path folds the pending decay into the sample-count check alone rather
	// than rewriting the whole table.
	if s.counts[idx]*s.decayFactorLocked() < minTurnSamples {
		return 0, false
	}
	var tail float64
	for k := idx; k < maxTrackedTurns; k++ {
		tail += s.counts[k]
	}
	return tail / s.counts[idx], true
}

// decayFactorLocked returns the multiplier owed since the last decay.
func (s *turnSurvival) decayFactorLocked() float64 {
	if s.halfLife <= 0 {
		return 1
	}
	elapsed := s.now().Sub(s.anchor)
	if elapsed <= 0 {
		return 1
	}
	return math.Exp2(-elapsed.Seconds() / s.halfLife.Seconds())
}

// decayLocked ages every count. Applied lazily on write, following the scheme
// lasState uses for attained service, so nothing runs on a timer.
func (s *turnSurvival) decayLocked() {
	factor := s.decayFactorLocked()
	if factor == 1 {
		return
	}
	for i := range s.counts {
		s.counts[i] *= factor
	}
	s.anchor = s.now()
}
