# Third-party licenses

This license index adapts the SPARC3D third-party notices for Hi3D v1.0. The
original license texts and attributions are retained, with additional references
for separately obtained models and optional examples.

The original SPARC3D license is retained in [Original Apache license](#original-apache-license).
Hi3D-owned or otherwise licensable materials identified in Schedule 2 are governed
by the Hi3D Agreement in [LICENSE](LICENSE). See [License scope](#license-scope)
for the component boundaries; the new Agreement does not replace original
third-party licenses or previously granted rights.

Parts of this package are derived from other open-source projects. Each is listed
below with the files it covers and the full text of its license, as those licenses
require. Every file named here also carries a provenance header pointing back to
this document.

| Upstream | License | Covers |
| --- | --- | --- |
| [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) | MIT | the sparse transformer and its building blocks |
| [Pointcept / Point Transformer V3](https://github.com/Pointcept/Pointcept) | MIT | the `Point` structure and the sparse PTv3 decoder |
| [O-CNN (ocnn-pytorch)](https://github.com/octree-nn/ocnn-pytorch) | MIT | Z-order (Morton) encoding |
| [NVIDIA FlexiCubes](https://github.com/nv-tlabs/FlexiCubes) | Apache-2.0 | dual-marching-cubes surface extraction |

---

## License scope

This index explains the component boundaries in Schedule 2 of the
[Hi3D Open License Agreement Version 1.0](LICENSE). It does not amend the
Agreement or grant additional rights. The Agreement and each applicable
third-party license govern their respective materials.

| Component | Applicable terms |
| --- | --- |
| Second-stage implementation and modifications in `sparc3d/`, including package initialization | The Hi3D Agreement applies only to content OCEANO HK HOLDING LIMITED owns or is authorized to license; original third-party rights in the same files remain under their original terms |
| `demo/refine_mesh.py`, `demo/image_to_mesh.py`, `demo/coarse_stage.py` | The Hi3D Agreement applies to licensable example code; Hunyuan's separate implementation, weights and outputs are excluded |
| The README, documentation and configuration paths listed in Schedule 2 | The Hi3D Agreement applies to licensable content; original notices and third-party terms remain in effect |
| The eight refinement artifacts listed in Schedule 2 | Distributed separately from the source code and Python packages under the Hi3D Agreement; other files or versions are not included merely by sharing a repository |
| Original SPARC3D, TRELLIS, Pointcept, O-CNN and FlexiCubes rights | Retained Apache-2.0 or MIT terms, as detailed in this file and the original source headers |
| Hunyuan3D-2mini, DINOv2, CLIP and other external dependencies | Their own upstream licenses; not relicensed by Hi3D |
| The five files listed as Sample Media in Schedule 2 | Excluded from the Hi3D Agreement; their presence or display does not grant reuse rights |
| Unreleased first-stage materials, training code, datasets, and brand assets | Excluded, subject to the limited trademark permissions expressly granted in the Agreement |

`LicenseRef-Hi3D-1.0` denotes the custom Agreement in [LICENSE](LICENSE), not a
blanket license for every file. Previously granted rights under applicable
Apache-2.0 or MIT licenses are not withdrawn or replaced by the Hi3D Agreement.

The source archive contains media expressly excluded by Schedule 2. Therefore
its package metadata does not assert a single SPDX license expression for the
whole archive. The complete applicable agreements and notices are included with
the Python distributions.

---

## Microsoft TRELLIS

Covers:

- `sparc3d/models/transformers/sparc3d_transformer.py`
- `sparc3d/nn/norm.py`
- `sparc3d/nn/rope.py`
- `sparc3d/nn/utils.py`
- `sparc3d/nn/sparse/basic.py`
- `sparc3d/nn/sparse/conv.py`
- `sparc3d/nn/sparse/linear.py`
- `sparc3d/nn/sparse/nonlinearity.py`
- `sparc3d/nn/sparse/spatial.py`
- `sparc3d/nn/sparse/attention/full_attn.py`
- `sparc3d/nn/sparse/attention/modules.py`
- `sparc3d/nn/sparse/transformer/blocks.py`
- `sparc3d/nn/sparse/transformer/modulated.py`

Each of these files states the specific SPARC3D modification in its header. In
general the training paths, the alternative attention backends, and the dense
variants were removed, leaving the released inference architecture.

```
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```

---

## Pointcept / Point Transformer V3

Point Transformer V3 (Xiaoyang Wu et al.) is released as part of Pointcept.

Covers:

- `sparc3d/models/autoencoders/blocks.py`
- `sparc3d/models/autoencoders/ptv3.py`
- `sparc3d/structures/point.py`
- `sparc3d/structures/point_modules.py`
- `sparc3d/utils/misc.py`
- `sparc3d/utils/precision.py`
- `sparc3d/utils/serialization/default.py`

The decoder half was specialized for the sparse mesh autoencoder; the encoder,
the training paths, and the Hilbert serialization order were removed.

```
MIT License

Copyright (c) 2023 Pointcept

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## O-CNN (ocnn-pytorch)

Covers:

- `sparc3d/utils/serialization/z_order.py`

The Morton decode tables and the batch-index field of the key were removed; only
`xyz2key` encoding is retained.

```
MIT License

Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## NVIDIA FlexiCubes

Covers:

- `sparc3d/ops/flexicubes/`

FlexiCubes retains its Apache-2.0 license. Its copy of the
license ships in the package at `sparc3d/ops/flexicubes/LICENSE.txt`, satisfying
Apache-2.0 section 4(a); the NVIDIA copyright notices are retained in the source
files, satisfying 4(c); and `sparc3d/ops/flexicubes/flexicubes.py` carries a
header enumerating every SPARC3D modification, satisfying 4(b). Upstream ships no
`NOTICE` file, so 4(d) does not apply.

The vendored copy is inference-only: the tetrahedral output, the learned
interpolation weights, the developability regularizer, and the gradient hooks
were removed.


---

## External models and optional examples

This section indexes components obtained separately at runtime or by the user.
It does not add their code or weights to this distribution, grant a license to
them, or replace their upstream terms. Neither the Hi3D Agreement nor a retained
source-code license automatically applies to external model weights or
third-party outputs.

### Required external image encoders

The default image conditioner loads the following models separately. Neither is
included in the Hi3D refinement checkpoint or in this source tree.

| Component | Role | Official model source | Upstream license reference |
| --- | --- | --- | --- |
| DINOv2-large | Spatial image features | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) | [Apache-2.0, DINOv2 LICENSE](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) |
| CLIP ViT-L/14 | Global image conditioning | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | [MIT, CLIP LICENSE](https://github.com/openai/CLIP/blob/main/LICENSE) |

Consult each model's own model card and the terms supplied with the version you
obtain. Background-removal models used by the optional `rembg` integration are
also obtained separately and retain their own terms.

### Optional Hunyuan3D coarse-mesh example

`demo/coarse_stage.py` is a Hi3D / SPARC3D example adapter. It imports
`hy3dgen.shapegen.Hunyuan3DDiTFlowMatchingPipeline` only when that example is used.
Its default model is `tencent/Hunyuan3D-2mini`, subfolder
`hunyuan3d-dit-v2-mini-turbo`.

**Hunyuan3D is only an optional example of coarse-mesh generation.** It is not a
mandatory dependency of the Hi3D refinement library or a required product
integration. `demo/refine_mesh.py` and the core `sparc3d` package accept a supplied
coarse mesh without using Hunyuan3D. Other suitable coarse-mesh generators can be
used instead. Installing the `[inference]` or `[demo]` extras does not install
Hunyuan3D.

**Tencent's Hunyuan3D implementation and pretrained weights are third-party
releases, not open-source content released by Hi3D.** They are not bundled with
this repository and are not relicensed under the Hi3D Agreement or any
retained source-code license. Users who
choose this example obtain them separately and comply with their upstream terms.

| Component used by the example | Official source | Applicable upstream agreement |
| --- | --- | --- |
| Hunyuan3D-2 implementation (`hy3dgen`) | [Tencent-Hunyuan/Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) | [Tencent Hunyuan 3D 2.0 Community License Agreement — source LICENSE](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE) |
| Hunyuan3D-2mini weights, mini-turbo variant | [tencent/Hunyuan3D-2mini](https://huggingface.co/tencent/Hunyuan3D-2mini) | [Tencent Hunyuan 3D 2.0 Community License Agreement — model LICENSE](https://huggingface.co/tencent/Hunyuan3D-2mini/blob/main/LICENSE) |

Upstream license terms are version-specific. If you select a different model,
subfolder, or source version, consult its matching license and model card.
Example-only use does not waive applicable restrictions, attribution,
redistribution, or hosted-service obligations in those agreements.

### Hi3D refinement weights

The eight Hi3D refinement artifacts identified and released under Schedule 2
are governed by the Hi3D Agreement in [LICENSE](LICENSE). They are distributed
separately from this source repository; see [License scope](#license-scope).
Other model files are not covered merely because they share the repository or
use a similar name.


---

## Original Apache license

The following is the original SPARC3D Apache-2.0 text, retained without changes.
It continues to govern the materials and rights to which it originally applied;
the Hi3D Agreement does not revoke those grants.

```text

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2026 The SPARC3D Authors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
