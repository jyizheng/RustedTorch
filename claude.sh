#!/bin/bash

# Set a custom CARGO_HOME environment variable
export CARGO_HOME="$HOME/.cargo"
npx claude-flow@alpha swarm "convert cpp source code to rust library and make sure the code in tests still work" --strategy development --claude
