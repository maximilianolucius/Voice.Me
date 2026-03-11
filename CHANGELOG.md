# Changelog

## [0.22.0] - 2024-01-01

### Changed
- Rebranded from Coqui TTS to Voice.Me
- Consolidated XTTS v1/v2 recipes into single parametrized script
- Merged tts_tests2/ into tts_tests/
- Added module-level docstrings to all Python files
- Created pyproject.toml for modern Python packaging
- Removed CPML license prompt

### Fixed
- Fixed inverted logic in voiceme() dataset formatter
- Fixed bare except clauses in model manager
- Replaced eval() with ast.literal_eval() for security
- Fixed hardcoded development paths in recipes and configs

### Removed
- Removed Coqui Studio references
- Removed AGENT.md (unrelated project file)
