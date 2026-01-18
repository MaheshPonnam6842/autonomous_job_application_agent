## Design Decisions

### Why start with a baseline version?
To validate data flow, state contracts, and safety constraints before
introducing more complex reasoning and generation.

### Why scoped rewriting instead of full rewrite?
To reduce hallucination risk and ensure changes are explainable and reversible.

### Why separate decision logic from rewrite logic?
To make the system auditable and easier to adapt to different hiring scenarios.
