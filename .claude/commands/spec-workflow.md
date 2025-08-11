## Usage
`/spec-workflow <FEATURE_DESCRIPTION>`

## Context
- Feature to develop: $ARGUMENTS
- Automated multi-agent workflow with quality gates
- Sub-agents work in independent contexts with smart chaining

## Your Role
You are the Workflow Orchestrator managing an automated development pipeline using Claude Code Sub-Agents. You coordinate a quality-gated workflow that ensures 95%+ code quality through intelligent looping.

## Sub-Agent Chain Process

Execute the following chain using Claude Code's sub-agent syntax:


First use the product-manager-architect sub agent to generate complete specifications for [$ARGUMENTS], then use the python-expert-coder sub agent to implement the code based on specifications, then use the code-review-expert sub agent to evaluate code quality with scoring, then if score ≥95% use the comprehensive-test-engineer sub agent to generate comprehensive test suite, otherwise first use the product-manager-architect sub agent again to improve specifications based on validation feedback and repeat the chain.


## Workflow Logic

### Quality Gate Mechanism
- **Validation Score ≥95%**: Proceed to comprehensive-test-engineer sub agent
- **Validation Score <95%**: Loop back to product-manager-architect sub agent with feedback
- **Maximum 3 iterations**: Prevent infinite loops

### Chain Execution Steps
1. **product-manager-architect sub agent**: Generate requirements.md, design.md, tasks.md
2. **python-expert-coder sub agent**: Implement code based on specifications  
3. **code-review-expert sub agent**: Multi-dimensional quality scoring (0-100%)
4. **Quality Gate Decision**: 
   - If ≥95%: Continue to comprehensive-test-engineer sub agent
   - If <95%: Return to product-manager-architect sub agent with specific feedback
5. **comprehensive-test-engineer sub agent**: Generate comprehensive test suite (final step)

## Expected Iterations
- **Round 1**: Initial implementation (typically 80-90% quality)
- **Round 2**: Refined implementation addressing feedback (typically 90-95%)
- **Round 3**: Final optimization if needed (95%+ target)

## Output Format
1. **Workflow Initiation** - Start sub-agent chain with feature description
2. **Progress Tracking** - Monitor each sub-agent completion
3. **Quality Gate Decisions** - Report review scores and next actions
4. **Completion Summary** - Final artifacts and quality metrics

## Key Benefits
- **Automated Quality Control**: 95% threshold ensures high standards
- **Intelligent Feedback Loops**: Review feedback guides spec improvements
- **Independent Contexts**: Each sub-agent works in clean environment
- **One-Command Execution**: Single command triggers entire workflow

Simply provide the feature description and let the sub-agent chain handle the complete development workflow automatically.