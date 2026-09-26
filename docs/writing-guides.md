# Writing short, useful guides

Use these notes for Liesel-GAM guides and tutorial notebooks, including model
building, optimization, and MCMC sampling.

## Give each topic a home

Read the current docs and implementation, identify the reader's task, and choose
where the explanation belongs:

- **Landing page:** explain the purpose, show a small working example, and link
  to the next steps.
- **Tutorial:** walk through a complete, realistic task, introducing one idea at
  a time and showing the results.
- **Task guide:** answer a specific question, such as choosing priors,
  configuring a sampler, or inspecting results.
- **API reference:** document arguments, defaults, exact rules, and edge cases.

Give each explanation one main home and link to it elsewhere. Keep prerequisites
and essential constraints beside the example they affect. Explain choices that
change the model, algorithm, or reported results; brevity must not hide them.
When retiring duplicate notebooks, preserve useful examples in tutorials and
genuine checks in the test suite.

Put `Overview <self>` first in the first toctree of each guide landing page.
Use Sphinx's special `self` entry, not the page's filename, so Overview links to
the landing page without nested children. Preserve the remaining entries and
their order. Check the sidebar on both the landing page and its child pages.

## Lead with the main workflow

Introduce a feature's general purpose before relating it to special cases. Show
the main workflow and a useful result first. Put tuning, performance details,
and extended diagnostics later, or link to a separate task guide or reference.

Give each section a practical heading, a short introduction, the relevant code,
and an interpretation of the result. Name the actual operation, such as “Define
the model” or “Choose the source.” Avoid vague headings and explanations of every
visible line of code.

Keep page and section headings on one line in both rendered sidebars at desktop
widths. Use a short toctree label when a longer tutorial title is useful. Shorten
the wording without losing its meaning; do not hide or clip wrapping text.

## Keep the language human

Use short sentences, familiar verbs, and concrete names. Address the reader
directly and explain technical terms when they become necessary. Preserve exact
API names and distinctions that affect the result.

Call model parameters “parameters,” not “coordinates.” Where transformations
matter, use “transformed parameters” or specify the parameter scale.

Prefer “This saves memory” to “This configuration facilitates reduced memory
consumption.” Remove repeated introductions, promotional claims, and closing
summaries that restate the section.

## Make examples easy to use

Use executable MyST Markdown for guides with code and results. Put runnable
Python in `{code-cell}` blocks without interactive prompts (`>>>` or `...`).

- State prerequisites, such as an existing `model`, before a snippet. Make
  complete tutorials runnable from top to bottom.
- Use existing public helpers instead of manual setup or calculations that they
  already handle. Include a lower-level recipe or alternative only when it serves
  a distinct task or helps the reader make a meaningful choice.
- Keep related setup and modifications together. Give each inspection expression
  its own `{code-cell}`, with its native output immediately below.
- Format for visual readability, not just line length. Group code into meaningful
  stages, with blank lines between stages and substantial independent definitions.
  Keep short, closely related statements together. Use brief comments to label
  conceptual groups when the surrounding prose does not make them clear.
- Wrap dense calls so functions or lambdas, inputs, nested expressions, and named
  options are easy to distinguish. Use trailing commas so Ruff preserves the
  layout. Keep simple calls compact. Apply this judgment to all functions and
  constructors, including plotting helpers, distributions, and table construction.
- Pass distributions to `lsl.Var` and its factory methods with `dist=`, never as
  positional arguments or using the deprecated `distribution=` keyword.
- Pass a single model root directly, as in `lsl.Model(y)`. Choose `to_float32`
  for the needs of the example, independently of this calling style.
- Use realistic data, fixed seeds, and only the settings needed for the task.
- Prefer expressions over `print()`. Use native tables for related results.
  Select useful fields and round numbers for readability. Remove unnecessary
  inspection calls instead of leaving them without output.
- Keep verification assertions in tests. Investigate awkward API behavior before
  adding repeated defensive checks to examples.

Tag build-only prerequisite setup with `:tags: [remove-cell]`. These cells still
execute but display neither code, outputs, nor an expandable box. Do not use
`hide-input` or `hide-cell` for this purpose. Keep prerequisites, required variable
names, consequential model choices, and links to relevant tutorials visible in
prose. Keep instructional construction and fitting visible in complete tutorials.
Imports stay visible in a code cell at the top of each guide. Hide only
non-import build setup, such as logging configuration and fixture models or data.

For example, a task guide can state that `tb` is the demo-data builder from the
smooth-curve tutorial, then prepare it for the build with:

````markdown
```{code-cell} ipython3
import liesel_gam as gam
```

```{code-cell} ipython3
:tags: [remove-cell]

df = gam.demo_data(n=200, seed=1)
tb = gam.TermBuilder.from_df(df)
```
````

Migration guides should show the old and new code and explain meaningful
behavior changes.

## Show useful visuals

Prefer built-in Liesel/Goose plotting helpers, such as `model.plot()` and
`gs.plot_trace()`. Use plotnine when no suitable helper exists. Put each plotting
call in its own code cell, directly above its rendered figure. Keep construction
and fitting separate. Explain what readers should look for and what the plot
cannot establish. Use enough contrast,
distinct shapes, or small positional offsets to keep overlapping marks visible.

In model walkthroughs, include `model.plot()` in its own cell after constructing
the model and show the graph immediately below. The Read the Docs build installs
Graphviz for layout. Give every figure descriptive alt text. For cell outputs,
use `mystnb.image.alt` cell metadata and check that it appears on the rendered
image.

Embed interactive explanations beside the relevant text. A separate-page link
can supplement the embed. Keep essential explanations readable without
interacting with the visual.

Static assets can illustrate concepts outside the executable example. Record
their source and refresh them when the explanation changes.

## Check the finished result

Execute guide examples during the docs build and fail the build on execution
errors. Let execution produce the displayed results; do not maintain copied
output blocks by hand.

Read the guide from top to bottom for flow, missing prerequisites, repetition,
and unnecessary detours. Match verification to the change: execute changed
examples through the docs build and inspect their generated outputs and plots.
Check that build-only setup leaves no code, output, or expandable box. Inspect
the rendered examples, figures, and navigation, including both sidebars and links.
Run the relevant hooks on edited files.

Use Sphinx cross-references for internal pages and API objects. Links intended
for use outside the docs must work from that context. Report validation accurately,
including whether builds were fresh or incremental and notebooks were executed,
cached, or skipped.

## Execute maintained guides in the build

Start each executable `.md` guide with notebook front matter so MyST-NB
recognizes it and selects the intended kernel:

```yaml
---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---
```

Use MyST-NB with `nb_execution_mode = "force"`,
`nb_execution_allow_errors = False`, and `nb_execution_raise_on_error = True`.
Execution errors must fail the build. Do not replace failing cells with skipped
execution, hand-copied output, or an alternative rendering pipeline. Build with
`sphinx-build -b html -W --keep-going docs/source docs/build/html` using the
compatible Liesel environment. The notebook kernel must use that environment too.

For every figure, provide descriptive cell metadata such as:

````markdown
```{code-cell} ipython3
---
mystnb:
  image:
    alt: A normal response depends on an intercept and a centered smooth.
---
model.plot()
```
````

The `.ipynb` example-library archive is explicitly excluded from execution and
labeled as saved historical output. This exemption is not available to maintained
guides: migrate an example to MyST Markdown when promoting or substantially
rewriting it. Keep its docname when practical so existing links keep working.

The base branch requires the published `optim-base` API. The Laplace guide branch
also requires the published, unmerged `laplace-loss` source. Pin the compatible
Liesel commit in the docs requirements and hosted build configuration. Do not
silently fall back to the released API or disable execution when that dependency
is missing.
