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

package pinuntiloverload

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

type stubEndpoint struct {
	metadata *datalayer.EndpointMetadata
	metrics  *datalayer.Metrics
	attr     datalayer.AttributeMap
}

func newStubEndpoint(name string, waiting int) *stubEndpoint {
	return &stubEndpoint{
		metadata: &datalayer.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name, Namespace: "default"}},
		metrics:  &datalayer.Metrics{WaitingQueueSize: waiting},
		attr:     datalayer.NewAttributes(),
	}
}

func (f *stubEndpoint) GetMetadata() *datalayer.EndpointMetadata           { return f.metadata }
func (f *stubEndpoint) UpdateMetadata(*datalayer.EndpointMetadata)         {}
func (f *stubEndpoint) GetMetrics() *datalayer.Metrics                     { return f.metrics }
func (f *stubEndpoint) UpdateMetrics(*datalayer.Metrics)                   {}
func (f *stubEndpoint) GetAttributes() datalayer.AttributeMap              { return f.attr }
func (f *stubEndpoint) String() string                                     { return f.metadata.ID.String() }
func (f *stubEndpoint) Put(key fwkplugin.DataKey, val datalayer.Cloneable) { f.attr.Put(key, val) }
func (f *stubEndpoint) Get(key fwkplugin.DataKey) (datalayer.Cloneable, bool) {
	return f.attr.Get(key)
}
func (f *stubEndpoint) Keys() []fwkplugin.DataKey     { return f.attr.Keys() }
func (f *stubEndpoint) Clone() datalayer.AttributeMap { return f.attr.Clone() }

func endpointNames(endpoints []scheduling.Endpoint) []string {
	names := make([]string, 0, len(endpoints))
	for _, ep := range endpoints {
		names = append(names, ep.GetMetadata().ID.Name)
	}
	return names
}

// fakeClock is advanced by the test between Filter calls.
type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time          { return c.t }
func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }

func newTestFilter(t *testing.T, params parameters) (*PinUntilOverloadFilter, *fakeClock) {
	t.Helper()
	f, err := NewPinUntilOverloadFilter("", params)
	require.NoError(t, err)
	clk := &fakeClock{t: time.Unix(1000, 0)}
	f.now = clk.now
	return f, clk
}

func pods(waitingA, waitingB, waitingC int) []scheduling.Endpoint {
	return []scheduling.Endpoint{
		newStubEndpoint("pod-b", waitingB),
		newStubEndpoint("pod-a", waitingA),
		newStubEndpoint("pod-c", waitingC),
	}
}

func TestFilter_PinsToFirstEndpointByNameWhileArmed(t *testing.T) {
	f, _ := newTestFilter(t, parameters{WaitingThreshold: 20, HoldSeconds: 30})

	got := f.Filter(context.Background(), nil, pods(0, 0, 0))

	assert.Equal(t, []string{"pod-a"}, endpointNames(got))
	assert.False(t, f.Released())
}

func TestFilter_PinsToConfiguredTarget(t *testing.T) {
	f, _ := newTestFilter(t, parameters{TargetEndpoint: "pod-c", WaitingThreshold: 20, HoldSeconds: 30})

	got := f.Filter(context.Background(), nil, pods(0, 0, 0))

	assert.Equal(t, []string{"pod-c"}, endpointNames(got))
}

func TestFilter_KeepsTargetChosenOnFirstRequest(t *testing.T) {
	f, _ := newTestFilter(t, parameters{WaitingThreshold: 20, HoldSeconds: 30})
	f.Filter(context.Background(), nil, []scheduling.Endpoint{newStubEndpoint("pod-b", 0)})

	got := f.Filter(context.Background(), nil, pods(0, 0, 0))

	assert.Equal(t, []string{"pod-b"}, endpointNames(got))
}

func TestFilter_ReleasesAfterHoldAndStaysReleased(t *testing.T) {
	f, clk := newTestFilter(t, parameters{WaitingThreshold: 20, HoldSeconds: 30})
	ctx := context.Background()

	assert.Equal(t, []string{"pod-a"}, endpointNames(f.Filter(ctx, nil, pods(20, 0, 0))))
	clk.advance(29 * time.Second)
	assert.Equal(t, []string{"pod-a"}, endpointNames(f.Filter(ctx, nil, pods(25, 0, 0))))
	clk.advance(1 * time.Second)
	assert.Equal(t, []string{"pod-b", "pod-a", "pod-c"}, endpointNames(f.Filter(ctx, nil, pods(25, 0, 0))))
	assert.True(t, f.Released())

	clk.advance(time.Minute)
	assert.Equal(t, []string{"pod-b", "pod-a", "pod-c"}, endpointNames(f.Filter(ctx, nil, pods(0, 0, 0))))
}

func TestFilter_DipBelowThresholdResetsHold(t *testing.T) {
	f, clk := newTestFilter(t, parameters{WaitingThreshold: 20, HoldSeconds: 30})
	ctx := context.Background()

	f.Filter(ctx, nil, pods(20, 0, 0))
	clk.advance(20 * time.Second)
	f.Filter(ctx, nil, pods(19, 0, 0))
	clk.advance(20 * time.Second)
	f.Filter(ctx, nil, pods(20, 0, 0))
	clk.advance(29 * time.Second)

	assert.Equal(t, []string{"pod-a"}, endpointNames(f.Filter(ctx, nil, pods(20, 0, 0))))
	assert.False(t, f.Released())
}

func TestFilter_ZeroHoldReleasesOnFirstOverload(t *testing.T) {
	f, _ := newTestFilter(t, parameters{WaitingThreshold: 5})

	got := f.Filter(context.Background(), nil, pods(5, 0, 0))

	assert.Len(t, got, 3)
	assert.True(t, f.Released())
}

func TestFilter_MissingTargetPassesThroughWithoutReleasing(t *testing.T) {
	f, _ := newTestFilter(t, parameters{TargetEndpoint: "pod-x", WaitingThreshold: 20, HoldSeconds: 30})

	got := f.Filter(context.Background(), nil, pods(0, 0, 0))

	assert.Len(t, got, 3)
	assert.False(t, f.Released())
}

func TestFactory(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		wantErr bool
	}{
		{name: "valid", raw: `{"waitingThreshold": 20, "holdSeconds": 30, "targetEndpoint": "pod-a"}`},
		{name: "missing threshold", raw: `{"holdSeconds": 30}`, wantErr: true},
		{name: "negative hold", raw: `{"waitingThreshold": 20, "holdSeconds": -1}`, wantErr: true},
		{name: "unknown field", raw: `{"waitingThreshold": 20, "bogus": 1}`, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Factory("pin", json.NewDecoder(bytes.NewBufferString(tt.raw)), nil)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
