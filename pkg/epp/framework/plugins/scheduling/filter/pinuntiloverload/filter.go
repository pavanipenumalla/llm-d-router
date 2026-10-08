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

// Package pinuntiloverload provides a benchmarking filter that sends every
// request to one endpoint until that endpoint's waiting queue stays at or
// above a threshold, and then passes all endpoints through for the rest of
// the process lifetime.
package pinuntiloverload

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// PinUntilOverloadFilterType is the type of the PinUntilOverloadFilter.
const PinUntilOverloadFilterType = "pin-until-overload-filter"

type parameters struct {
	// TargetEndpoint is the endpoint name to pin to. When empty, the first
	// candidate by name on the first request is chosen and kept.
	TargetEndpoint string `json:"targetEndpoint,omitempty"`

	// WaitingThreshold is the target's waiting queue size at which the hold
	// timer starts.
	WaitingThreshold int `json:"waitingThreshold"`

	// HoldSeconds is how long the waiting queue must stay at or above
	// WaitingThreshold before the filter releases. Zero releases on the first
	// request that sees the threshold reached.
	HoldSeconds int `json:"holdSeconds"`
}

var _ scheduling.Filter = &PinUntilOverloadFilter{}

// Factory defines the factory function for PinUntilOverloadFilter.
func Factory(name string, rawParameters *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	params := parameters{}
	if rawParameters != nil {
		rawParameters.DisallowUnknownFields()
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", PinUntilOverloadFilterType, err)
		}
	}
	return NewPinUntilOverloadFilter(name, params)
}

// NewPinUntilOverloadFilter validates the given parameters and returns a new
// PinUntilOverloadFilter with the given name.
func NewPinUntilOverloadFilter(name string, params parameters) (*PinUntilOverloadFilter, error) {
	if name == "" {
		name = PinUntilOverloadFilterType
	}
	if params.WaitingThreshold <= 0 {
		return nil, errors.New("pin-until-overload filter requires a positive waitingThreshold")
	}
	if params.HoldSeconds < 0 {
		return nil, fmt.Errorf("pin-until-overload filter requires a non-negative holdSeconds, got %d", params.HoldSeconds)
	}
	return &PinUntilOverloadFilter{
		typedName:        plugin.TypedName{Type: PinUntilOverloadFilterType, Name: name},
		target:           params.TargetEndpoint,
		waitingThreshold: params.WaitingThreshold,
		hold:             time.Duration(params.HoldSeconds) * time.Second,
		now:              time.Now,
	}, nil
}

// PinUntilOverloadFilter creates a load imbalance on purpose for experiments.
// While armed it keeps only the target endpoint. On each request it reads the
// target's model-server waiting queue size; once that stays at or above
// waitingThreshold for the hold duration, the filter releases and returns its
// input unchanged from then on. The release is one-way and in memory, so an
// EPP restart re-arms it.
type PinUntilOverloadFilter struct {
	typedName        plugin.TypedName
	waitingThreshold int
	hold             time.Duration
	now              func() time.Time

	mu        sync.Mutex
	target    string
	overSince time.Time
	released  bool
}

// TypedName returns the typed name of the plugin.
func (f *PinUntilOverloadFilter) TypedName() plugin.TypedName {
	return f.typedName
}

// Released reports whether the filter has stopped pinning.
func (f *PinUntilOverloadFilter) Released() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.released
}

// Filter returns only the target endpoint while armed and every endpoint once
// released. When the target is not among the candidates it returns them all
// without releasing.
func (f *PinUntilOverloadFilter) Filter(ctx context.Context, _ *scheduling.InferenceRequest,
	endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.released || len(endpoints) == 0 {
		return endpoints
	}

	logger := log.FromContext(ctx)
	if f.target == "" {
		f.target = firstByName(endpoints)
		logger.Info("Pin-until-overload filter chose target", "target", f.target)
	}

	var target scheduling.Endpoint
	for _, ep := range endpoints {
		if ep.GetMetadata().ID.Name == f.target {
			target = ep
			break
		}
	}
	if target == nil {
		logger.V(logutil.DEBUG).Info("Pin-until-overload target not among candidates, passing all through",
			"target", f.target, "candidates", len(endpoints))
		return endpoints
	}

	waiting, kvUsage := 0, 0.0
	if m := target.GetMetrics(); m != nil {
		waiting, kvUsage = m.WaitingQueueSize, m.KVCacheUsagePercent
	}

	now := f.now()
	if waiting < f.waitingThreshold {
		f.overSince = time.Time{}
		return []scheduling.Endpoint{target}
	}
	if f.overSince.IsZero() {
		f.overSince = now
	}
	if now.Sub(f.overSince) < f.hold {
		return []scheduling.Endpoint{target}
	}

	f.released = true
	logger.Info("Pin-until-overload filter released",
		"target", f.target, "waiting", waiting, "kvCacheUsage", kvUsage,
		"overloadedSince", f.overSince, "releasedAt", now)
	return endpoints
}

func firstByName(endpoints []scheduling.Endpoint) string {
	names := make([]string, 0, len(endpoints))
	for _, ep := range endpoints {
		names = append(names, ep.GetMetadata().ID.Name)
	}
	sort.Strings(names)
	return names[0]
}
