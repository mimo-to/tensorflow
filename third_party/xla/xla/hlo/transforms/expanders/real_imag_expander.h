/* Copyright 2021 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_HLO_TRANSFORMS_EXPANDERS_REAL_IMAG_EXPANDER_H_
#define XLA_HLO_TRANSFORMS_EXPANDERS_REAL_IMAG_EXPANDER_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"

namespace xla {

// Expands real/image instructions with non-complex inputs.
class RealImagExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "real_imag_expander"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* inst) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* inst) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_EXPANDERS_REAL_IMAG_EXPANDER_H_
