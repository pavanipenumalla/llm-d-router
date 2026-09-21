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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// newTestSurvival returns a table on a clock the caller advances, so decay is
// exercised deliberately rather than by however long a test takes to run.
func newTestSurvival() (*turnSurvival, *time.Time) {
	clock := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	s := &turnSurvival{anchor: clock, halfLife: turnHistoryHalfLife}
	s.now = func() time.Time { return clock }
	return s, &clock
}

// observeSession records the observations a session of the given length
// produces: one at each turn index it reaches.
func observeSession(s *turnSurvival, length int64) {
	for turn := int64(0); turn < length; turn++ {
		s.observe(turn)
	}
}

// The worked example from the proposal: one session of 1 turn and one of 5, a
// true mean length of 3. Averaging the observed turn indices gives 1.67 and
// never converges to 3; counting them and dividing gives 3 exactly, along with
// the conditional value at every turn.
//
// The population is replayed four times only to clear minTurnSamples. The
// estimate is a ratio, so replaying it does not change any expected value.
func TestTurnSurvival_MeanResidualLifeMatchesWorkedExample(t *testing.T) {
	s, _ := newTestSurvival()
	for i := 0; i < 4; i++ {
		observeSession(s, 1)
		observeSession(s, 5)
	}

	// At turn 0 the estimate is the unconditional mean length.
	got, ok := s.estimate(0)
	assert.True(t, ok)
	assert.InDelta(t, 3.0, got, 1e-9, "mean length of a 1-turn and a 5-turn session")

	// Past turn 0 only the 5-turn session survives, so the expected remaining
	// count rises to 4 before falling away. An estimator assuming every session
	// runs the mean length would report 2 here.
	for _, tc := range []struct {
		turn int64
		want float64
	}{
		{turn: 1, want: 4},
		{turn: 2, want: 3},
		{turn: 3, want: 2},
		{turn: 4, want: 1},
	} {
		got, ok := s.estimate(tc.turn)
		assert.True(t, ok, "turn %d", tc.turn)
		assert.InDelta(t, tc.want, got, 1e-9, "turn %d", tc.turn)
	}
}

// Counts are increments, so the estimate does not depend on observation order
// and never needs to know that a session finished.
func TestTurnSurvival_IndependentOfArrivalOrder(t *testing.T) {
	sequential, _ := newTestSurvival()
	for i := 0; i < 4; i++ {
		observeSession(sequential, 1)
		observeSession(sequential, 5)
	}

	// The same two sessions interleaved turn by turn, as concurrent traffic
	// would arrive, with the 5-turn session still mid-conversation throughout.
	interleaved, _ := newTestSurvival()
	for i := 0; i < 4; i++ {
		interleaved.observe(0) // the 1-turn session, complete
		interleaved.observe(0) // the 5-turn session, turn 1 of 5
	}
	for i := 0; i < 4; i++ {
		for turn := int64(1); turn < 5; turn++ {
			interleaved.observe(turn)
		}
	}

	for turn := int64(0); turn < 5; turn++ {
		want, okWant := sequential.estimate(turn)
		got, okGot := interleaved.estimate(turn)
		assert.Equal(t, okWant, okGot, "turn %d", turn)
		assert.InDelta(t, want, got, 1e-9, "turn %d", turn)
	}
}

// The central claim: whether expected remaining turns rises or falls with age is
// a property of the traffic, not an assumption baked into the estimator.
func TestTurnSurvival_FollowsTheDistributionsShape(t *testing.T) {
	t.Run("heavy tail raises the estimate with age", func(t *testing.T) {
		// Equal numbers of sessions at 1, 2, 4, 8, 16 and 32 turns.
		s, _ := newTestSurvival()
		for _, length := range []int64{1, 2, 4, 8, 16, 32} {
			for i := 0; i < 8; i++ {
				observeSession(s, length)
			}
		}

		previous := 0.0
		for _, turn := range []int64{0, 1, 2, 4, 8, 16} {
			got, ok := s.estimate(turn)
			assert.True(t, ok, "turn %d", turn)
			assert.GreaterOrEqual(t, got, previous,
				"a session that has survived to turn %d should expect at least as many turns left", turn)
			previous = got
		}

		// A deep session expects more turns left than a new one, which an
		// estimator subtracting the turn count from a mean cannot express.
		atStart, _ := s.estimate(0)
		atDepth, _ := s.estimate(16)
		assert.Greater(t, atDepth, atStart)
	})

	t.Run("fixed length lowers the estimate with age", func(t *testing.T) {
		// Every session runs exactly 10 turns, so remaining turns must count
		// down: this is the case a mean-minus-turns estimator gets right.
		s, _ := newTestSurvival()
		for i := 0; i < 8; i++ {
			observeSession(s, 10)
		}

		for turn := int64(0); turn < 10; turn++ {
			got, ok := s.estimate(turn)
			assert.True(t, ok, "turn %d", turn)
			assert.InDelta(t, float64(10-turn), got, 1e-9, "turn %d", turn)
		}
	})
}

// The estimate is at least 1 for free: the denominator is also the first term of
// the numerator, so there is no floor to apply.
func TestTurnSurvival_EstimateNeverBelowOne(t *testing.T) {
	s, _ := newTestSurvival()
	for i := 0; i < minTurnSamples; i++ {
		observeSession(s, 3)
	}
	for turn := int64(0); turn < 3; turn++ {
		got, ok := s.estimate(turn)
		assert.True(t, ok)
		assert.GreaterOrEqual(t, got, 1.0, "turn %d", turn)
	}
}

func TestTurnSurvival_WithholdsEstimateBelowMinSamples(t *testing.T) {
	s, _ := newTestSurvival()

	// One short of the threshold at turn 0.
	for i := 0; i < minTurnSamples-1; i++ {
		s.observe(0)
	}
	_, ok := s.estimate(0)
	assert.False(t, ok, "%d observations should not be enough", minTurnSamples-1)

	s.observe(0)
	_, ok = s.estimate(0)
	assert.True(t, ok, "%d observations should be enough", minTurnSamples)

	// A turn index nothing has reached carries no estimate.
	_, ok = s.estimate(500)
	assert.False(t, ok)
}

func TestTurnSurvival_IgnoresNegativeTurns(t *testing.T) {
	s, _ := newTestSurvival()
	s.observe(-1)
	_, ok := s.estimate(-1)
	assert.False(t, ok)

	var total float64
	for _, count := range s.counts {
		total += count
	}
	assert.Zero(t, total, "a negative turn must not be recorded anywhere")
}

// Sessions deeper than the table land in its last slot, where the estimate
// saturates at 1 rather than reading out of bounds.
func TestTurnSurvival_SaturatesPastTheTableSize(t *testing.T) {
	s, _ := newTestSurvival()
	for i := 0; i < minTurnSamples; i++ {
		s.observe(maxTrackedTurns)
		s.observe(maxTrackedTurns + 5000)
	}
	assert.Equal(t, float64(2*minTurnSamples), s.counts[maxTrackedTurns-1])

	got, ok := s.estimate(maxTrackedTurns + 1)
	assert.True(t, ok)
	assert.InDelta(t, 1.0, got, 1e-9)
}

func TestTurnSurvival_DecayHalvesCountsPerHalfLife(t *testing.T) {
	s, clock := newTestSurvival()
	for i := 0; i < 8; i++ {
		s.observe(0)
	}
	assert.InDelta(t, 8.0, s.counts[0], 1e-9)

	// A write is what folds decay in, so advance the clock and observe again:
	// the eight prior observations halve, then the new one is added.
	*clock = clock.Add(turnHistoryHalfLife)
	s.observe(0)
	assert.InDelta(t, 5.0, s.counts[0], 1e-9, "8 halved to 4, plus the new observation")
}

// Decay multiplies every entry by the same factor, and the estimate is a ratio
// of sums of entries, so ageing the table cannot move the value. Only the
// balance between old and new observations changes, and whether enough evidence
// is left to report a value at all.
func TestTurnSurvival_DecayDoesNotMoveTheValue(t *testing.T) {
	// Enough sessions that three half-lives still leave the sample count above
	// minTurnSamples, isolating the value from the availability check below.
	s, clock := newTestSurvival()
	for i := 0; i < 64; i++ {
		observeSession(s, 5)
	}
	before, ok := s.estimate(0)
	assert.True(t, ok)
	assert.InDelta(t, 5.0, before, 1e-9)

	// Decay owed but not yet folded in: the read path applies it to the sample
	// check only, so the value is unchanged.
	*clock = clock.Add(3 * turnHistoryHalfLife)
	pending, ok := s.estimate(0)
	assert.True(t, ok)
	assert.InDelta(t, before, pending, 1e-9)

	// Decay folded into every entry: unchanged again.
	s.decayLocked()
	folded, ok := s.estimate(0)
	assert.True(t, ok)
	assert.InDelta(t, before, folded, 1e-9)
}

// Decay is what makes evidence expire. Once a table has aged past the point
// where any turn index still holds minTurnSamples, the estimate is withdrawn and
// the caller falls back rather than being served a stale prior.
func TestTurnSurvival_WithdrawsEstimateOnceEvidenceHasAged(t *testing.T) {
	s, clock := newTestSurvival()
	for i := 0; i < 8; i++ {
		observeSession(s, 5)
	}
	_, ok := s.estimate(0)
	assert.True(t, ok)

	// Three half-lives take 8 observations to 1, below the threshold.
	*clock = clock.Add(3 * turnHistoryHalfLife)
	_, ok = s.estimate(0)
	assert.False(t, ok, "evidence this old should no longer be trusted")
}
