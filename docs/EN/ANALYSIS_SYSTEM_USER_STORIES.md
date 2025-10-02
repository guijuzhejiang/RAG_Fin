# 📋 Analysis System - User Stories

## Overview

This document outlines user stories for the Amafi Analysis System, which leverages Independent Expert Reports (IERs) to provide investment banking-quality insights while maintaining complete data privacy through self-hosted open-source models.

**Key Architecture Principles**:
- **Data Privacy First**: All sensitive client data processed on-premise or in private cloud using open-source models (Qwen 72B, Llama 3.1 70B)
- **Direct Context Feeding**: Metadata-based retrieval of complete IER sections (not RAG) for MVP
- **Expert Knowledge Extraction**: Focus on extracting expert-curated comparable companies, valuation frameworks, and analytical rationale
- **Hybrid Future**: Layer RAG for broader datasets (broker research, news) in Phase 2

## Epic 1: Target Input & IER Discovery

### AS-001: Input Target Details
**As an** M&A advisor  
**I want to** input my target company's industry and geography  
**So that** the system can find relevant IERs for my analysis

#### Acceptance Criteria
- [ ] Can select industry from hierarchical tree (NAICS-based)
- [ ] Can select geography from structured tree (Country > Region > State)
- [ ] Size and EBITDA fields are optional
- [ ] Can specify transaction type (acquisition, merger, IPO)
- [ ] System validates minimum required fields (industry + geography)
- [ ] Can save target profile for future use

#### Priority: MUST HAVE
#### Story Points: 5

---

### AS-002: Find Relevant IERs
**As an** analyst  
**I want the** system to automatically find the most relevant IERs  
**So that** I don't have to manually search through hundreds of documents

#### Acceptance Criteria
- [ ] System searches entire IER database based on input
- [ ] Returns ranked list of relevant IERs (top 20-30)
- [ ] Shows relevance score for each IER (0-100%)
- [ ] Displays key match factors (industry ✓, geography ✓, size ~)
- [ ] Can filter results by date range
- [ ] Can adjust relevance criteria weights

#### Priority: MUST HAVE
#### Story Points: 8

---

### AS-003: Handle Unknown Size Parameters
**As a** user without complete information  
**I want to** get analysis even without knowing target size/EBITDA  
**So that** I can still receive valuable insights

#### Acceptance Criteria
- [ ] System works with just industry + geography
- [ ] Provides size distribution analysis from similar IERs
- [ ] Shows typical size ranges in the industry
- [ ] Returns broader set of IERs (30 vs 20) when size unknown
- [ ] Indicates confidence level based on available data

#### Priority: MUST HAVE
#### Story Points: 5

---

## Epic 2: Section Detection & Extraction Foundation

### AS-021: Smart Section Boundary Detection
**As a** system
**I need to** reliably detect section boundaries in IERs
**So that** I can extract the correct content for analysis

#### Acceptance Criteria
- [ ] Primary: TOC-based section detection with 90%+ accuracy
- [ ] Secondary: Keyword-based detection when TOC unavailable
- [ ] Tertiary: Position-based heuristics as fallback
- [ ] Quaternary: Content pattern matching as last resort
- [ ] Validates section length (3-15 pages for Industry Overview)
- [ ] Returns confidence score with each detection
- [ ] Handles variations in section naming

#### Dependencies
- None (foundation story)

#### Priority: MUST HAVE
#### Story Points: 8

---

### AS-022: LLM Processing Pipeline
**As a** system
**I need to** efficiently process multiple IER sections through LLM
**So that** I can minimize costs and latency

#### Acceptance Criteria
- [ ] Batch process up to 5 IERs in parallel
- [ ] Implement prompt caching for industry context (40% cost savings)
- [ ] Retry logic with exponential backoff for API failures
- [ ] Cache IER extractions for 30 days
- [ ] Track token usage and costs per analysis
- [ ] Handle LLM rate limits gracefully with queuing
- [ ] Process time < 15 seconds per IER

#### Dependencies
- None (foundation story)

#### Priority: MUST HAVE
#### Story Points: 13

---

## Epic 3: Industry Overview Analysis

### AS-004: Extract Industry Overview Section
**As an** analyst  
**I want the** system to extract and analyze the Industry Overview section  
**So that** I can quickly understand market dynamics

#### Acceptance Criteria
- [ ] Automatically locates Industry Overview section (8-10 pages)
- [ ] Extracts market size and growth rates
- [ ] Identifies top players and market shares
- [ ] Extracts competitive dynamics
- [ ] Identifies growth drivers and challenges
- [ ] Processing time < 30 seconds per IER

#### Priority: MUST HAVE
#### Story Points: 13

---

### AS-005: Aggregate Industry Insights
**As a** decision maker  
**I want to** see aggregated insights from multiple IERs  
**So that** I can understand industry consensus views

#### Acceptance Criteria
- [ ] Combines insights from 10-20 relevant IERs
- [ ] Shows median/mean market metrics
- [ ] Identifies common themes across IERs
- [ ] Highlights outliers or contrarian views
- [ ] Provides statistical confidence levels
- [ ] Generates industry summary dashboard

#### Priority: MUST HAVE
#### Story Points: 8

---

### AS-006: Industry-Specific Risk Analysis
**As a** risk manager  
**I want to** understand industry-specific risks  
**So that** I can properly assess the deal

#### Acceptance Criteria
- [ ] Extracts risk factors from Industry Overview sections
- [ ] Categorizes risks (regulatory, competitive, technology, market)
- [ ] Shows frequency of risk mentions across IERs
- [ ] Highlights emerging risks (newly appearing in recent IERs)
- [ ] Maps risks to mitigation strategies mentioned

#### Priority: SHOULD HAVE
#### Story Points: 8

---

## Epic 3: Valuation Analysis

### AS-007: Extract Expert-Curated Comparable Companies
**As an** investment banker
**I want to** access pre-vetted comparable companies from IER valuation sections
**So that** I can skip hours of manual comp research and use expert judgment

#### Acceptance Criteria
- [ ] Extracts 10-15 comparable companies per IER valuation section
- [ ] Captures expert rationale (1-2 paragraphs per company explaining comparability)
- [ ] Preserves intelligent segmentation (Tier 1 pure-plays, Tier 2 regional, Tier 3 global)
- [ ] Stores company tickers for live data refresh
- [ ] Extracts adjustment framework (size discount, geographic premium, etc.)
- [ ] Captures expert commentary on why each comp is relevant
- [ ] Stores in structured format for programmatic access

#### Example Output:
```json
{
  "company_name": "Company X Ltd",
  "ticker": "ASX:XYZ",
  "segment": "Tier 1: Australian Pure-Plays",
  "expert_rationale": "Leading manufacturer of surgical instruments with similar product portfolio...",
  "key_differences": "Larger scale (2.5x revenue), adjust down by 0.5x multiple...",
  "adjustments": {"size_discount": -0.5, "geographic_premium": 0.3}
}
```

#### Priority: MUST HAVE (Core Value Proposition)
#### Story Points: 13

---

### AS-007b: Refresh Comparables with Live Market Data
**As an** analyst
**I want to** refresh expert-selected comparables with current market prices
**So that** I get current multiples based on expert comp selection

#### Acceptance Criteria
- [ ] Fetches live market data for all comp tickers (share price, market cap, EV)
- [ ] Calculates current multiples (EV/EBITDA, EV/Revenue, P/E)
- [ ] Applies expert segmentation weighting (Tier 1: 50%, Tier 2: 30%, Tier 3: 20%)
- [ ] Uses expert adjustment framework for target-specific factors
- [ ] Shows valuation as of current date
- [ ] Compares to IER date multiples (historical vs current)
- [ ] Handles missing/delisted companies gracefully

#### Dependencies
- AS-007 (Extract Comparable Companies)
- Market data API integration (Capital IQ, Bloomberg, or alternative)

#### Priority: MUST HAVE
#### Story Points: 8

---

### AS-007c: Extract Precedent Transactions
**As an** investment banker
**I want to** access precedent transactions from IERs
**So that** I can reference historical deal multiples and premiums

#### Acceptance Criteria
- [ ] Extracts 5-10 precedent transactions per IER
- [ ] Captures: deal value, multiples (EV/EBITDA, EV/Revenue), premium paid
- [ ] Records buyer type (strategic vs financial)
- [ ] Stores deal rationale and synergies disclosed
- [ ] Captures expert commentary on transaction relevance
- [ ] No live data needed (historically accurate as of IER date)
- [ ] Builds comprehensive database across all processed IERs

#### Priority: MUST HAVE
#### Story Points: 8

---

### AS-008: Generate Valuation Football Field
**As an** analyst  
**I want to** see a football field chart of valuations  
**So that** I can visualize the valuation range

#### Acceptance Criteria
- [ ] Auto-generates football field from multiple IERs
- [ ] Shows different methodology ranges
- [ ] Indicates proposed transaction value
- [ ] Highlights median and quartiles
- [ ] Exportable to PowerPoint/Excel
- [ ] Interactive (can exclude outliers)

#### Priority: SHOULD HAVE
#### Story Points: 13

---

### AS-009: Multiple Benchmarking
**As a** valuation specialist  
**I want to** benchmark multiples against the market  
**So that** I can justify my valuation

#### Acceptance Criteria
- [ ] Shows distribution of multiples from relevant IERs
- [ ] Segments by time period (LTM, NTM)
- [ ] Filters by deal size ranges
- [ ] Shows premium/discount to median
- [ ] Provides statistical analysis (std dev, percentiles)
- [ ] Generates benchmarking report

#### Dependencies
- AS-004 (Extract Industry Overview)
- AS-021 (Smart Section Detection)

#### Priority: MUST HAVE
#### Story Points: 8

---

### AS-024: Deal Similarity Scoring
**As an** analyst
**I want to** see how similar each IER is to my target
**So that** I can assess relevance and prioritize which IERs to review deeply

#### Acceptance Criteria
- [ ] Similarity score (0-100%) for each IER
- [ ] Breakdown by: industry match, geography, size, recency
- [ ] Visual indicator (color-coded badges)
- [ ] Can sort/filter by similarity score
- [ ] Explains why IER is or isn't similar
- [ ] Adjusts weights when size is unknown

#### Dependencies
- AS-002 (Find Relevant IERs)

#### Priority: SHOULD HAVE
#### Story Points: 5

---

## Epic 5: Report Generation

### AS-010: Generate Buy-Side Advisory Report
**As a** buy-side advisor  
**I want to** generate a comprehensive advisory report  
**So that** I can advise my client on the acquisition

#### Acceptance Criteria
- [ ] Generates 15-20 page PDF report
- [ ] Includes executive summary with go/no-go recommendation
- [ ] Contains industry analysis section
- [ ] Shows valuation benchmarking
- [ ] Identifies key risks and opportunities
- [ ] Includes source IER references
- [ ] Customizable template

#### Dependencies
- AS-005 (Aggregate Industry Insights)
- AS-007 (Extract Valuation Summary)
- AS-009 (Multiple Benchmarking)

#### Priority: MUST HAVE
#### Story Points: 13

---

### AS-011: Generate Sell-Side Positioning Report
**As a** sell-side advisor  
**I want to** generate a positioning report  
**So that** I can maximize value for my client

#### Acceptance Criteria
- [ ] Identifies value drivers to emphasize
- [ ] Shows premium justification from comparables
- [ ] Suggests optimal deal structure
- [ ] Identifies likely buyers
- [ ] Provides negotiation leverage points
- [ ] Includes market timing analysis

#### Dependencies
- AS-005 (Aggregate Industry Insights)
- AS-007 (Extract Valuation Summary)

#### Priority: SHOULD HAVE
#### Story Points: 13

---

### AS-012: Export to Excel Model
**As a** financial analyst  
**I want to** export data to Excel  
**So that** I can build my own models

#### Acceptance Criteria
- [ ] Exports all metrics to structured Excel
- [ ] Includes formulas and links
- [ ] Maintains data relationships
- [ ] Includes source documentation
- [ ] Compatible with standard models
- [ ] Preserves formatting

#### Dependencies
- AS-009 (Multiple Benchmarking)

#### Priority: MUST HAVE
#### Story Points: 5

---

## Epic 6: Interactive Analysis & UX

### AS-013a: Basic Question Answering
**As a** user
**I want to** ask specific questions about the IERs
**So that** I can get targeted insights

#### Acceptance Criteria
- [ ] Natural language question input (text box)
- [ ] System searches relevant IER sections
- [ ] Provides answers with confidence scores (0-100%)
- [ ] Shows source IER references
- [ ] Handles 6-8 common question patterns
- [ ] Response time < 30 seconds

#### Example Questions:
- "What EBITDA multiple should we pay?"
- "What are the main risks in medical devices?"
- "What synergies are typically achieved?"
- "How long do integrations usually take?"

#### Dependencies
- AS-004 (Extract Industry Overview)
- AS-022 (LLM Processing Pipeline)

#### Priority: SHOULD HAVE
#### Story Points: 8

---

### AS-013b: Follow-Up Questions
**As a** user
**I want to** ask follow-up questions based on previous answers
**So that** I can explore topics more deeply

#### Acceptance Criteria
- [ ] Maintains conversation context (up to 5 turns)
- [ ] References previous questions/answers
- [ ] Can drill down into specific IERs
- [ ] Can ask for examples or clarification
- [ ] Session history saved for 24 hours

#### Dependencies
- AS-013a (Basic Question Answering)

#### Priority: NICE TO HAVE
#### Story Points: 8

---

### AS-013c: Learning from User Feedback
**As a** system
**I want to** learn from user feedback on answers
**So that** I can improve answer quality over time

#### Acceptance Criteria
- [ ] Thumbs up/down on each answer
- [ ] Optional feedback text
- [ ] Tracks feedback metrics
- [ ] Adjusts answer ranking based on feedback
- [ ] Admin dashboard to review feedback

#### Dependencies
- AS-013a (Basic Question Answering)

#### Priority: NICE TO HAVE
#### Story Points: 5

---

### AS-014: Compare Specific IERs
**As an** analyst  
**I want to** compare 2-3 specific IERs side-by-side  
**So that** I can understand differences

#### Acceptance Criteria
- [ ] Select 2-3 IERs for comparison
- [ ] Shows side-by-side metrics
- [ ] Highlights key differences
- [ ] Compares methodologies used
- [ ] Shows opinion variations
- [ ] Exportable comparison table

#### Dependencies
- AS-002 (Find Relevant IERs)

#### Priority: NICE TO HAVE
#### Story Points: 8

---

### AS-025: Save Analysis as Template
**As a** frequent user
**I want to** save analysis configurations as templates
**So that** I can quickly run similar analyses

#### Acceptance Criteria
- [ ] Save search criteria as named template
- [ ] Include: industry, geography, filters, output format
- [ ] List of user's saved templates
- [ ] Edit and delete templates
- [ ] Share templates with team members
- [ ] Default templates for common industries

#### Dependencies
- AS-001 (Input Target Details)

#### Priority: SHOULD HAVE
#### Story Points: 5

---

### AS-026: Real-Time Progress Indicator
**As a** user waiting for analysis
**I want to** see processing progress in real-time
**So that** I know the system is working and estimate wait time

#### Acceptance Criteria
- [ ] Progress bar with percentage complete
- [ ] Stage indicators (Discovery → Extraction → Analysis → Report)
- [ ] Current action displayed ("Processing IER 3 of 18...")
- [ ] Estimated time remaining
- [ ] Can run analysis in background and get notified
- [ ] Can cancel long-running analysis

#### Dependencies
- AS-002 (Find Relevant IERs)

#### Priority: MUST HAVE
#### Story Points: 5

---

### AS-027: Bookmark and Annotate IERs
**As an** analyst
**I want to** bookmark interesting IERs and add notes
**So that** I can reference them later and share insights with team

#### Acceptance Criteria
- [ ] Bookmark/favorite individual IERs
- [ ] Add private notes to any IER
- [ ] Tag IERs with custom labels
- [ ] View all bookmarked IERs
- [ ] Search within bookmarks and notes
- [ ] Export bookmarks with notes

#### Dependencies
- AS-002 (Find Relevant IERs)

#### Priority: NICE TO HAVE
#### Story Points: 5

---

### AS-015: Track Analysis History
**As a** team lead  
**I want to** see what analyses have been performed  
**So that** I can avoid duplicate work

#### Acceptance Criteria
- [ ] Shows history of all analyses
- [ ] Searchable by target, user, date
- [ ] Can rerun previous analyses
- [ ] Can share analyses with team
- [ ] Shows version history
- [ ] Includes audit trail

#### Dependencies
- AS-010 (Generate Buy-Side Report)

#### Priority: SHOULD HAVE
#### Story Points: 5

---

### AS-028: Analysis Comparison
**As a** team lead
**I want to** compare two analyses side-by-side
**So that** I can see how valuations or insights changed over time

#### Acceptance Criteria
- [ ] Select 2 previous analyses to compare
- [ ] Side-by-side view of key metrics
- [ ] Highlights differences (increased/decreased)
- [ ] Shows what IERs were added/removed
- [ ] Can compare across different targets
- [ ] Export comparison report

#### Dependencies
- AS-015 (Track Analysis History)

#### Priority: NICE TO HAVE
#### Story Points: 8

---

## Epic 6.5: Data Privacy & Open-Source Model Management

### AS-045: Self-Hosted Model Deployment
**As an** IT administrator
**I want to** deploy open-source models on our private infrastructure
**So that** sensitive client data never leaves our organization

#### Acceptance Criteria
- [ ] Support for Qwen 2.5 72B (128K context) deployment
- [ ] Support for Llama 3.1 70B deployment
- [ ] Deploy on AWS VPC, Azure Private Cloud, or on-premise GPU servers
- [ ] Model serving via vLLM or TensorRT for optimized inference
- [ ] Health monitoring and auto-restart on failures
- [ ] GPU utilization monitoring dashboard
- [ ] Support for A100 (80GB) or 4x A10G (24GB each) configurations

#### Priority: MUST HAVE (Core Architecture)
#### Story Points: 21

---

### AS-046: Sensitive Data Isolation
**As a** compliance officer
**I want to** ensure sensitive client data is never sent to external APIs
**So that** we meet regulatory and confidentiality requirements

#### Acceptance Criteria
- [ ] All client financial data processed locally (no external API calls)
- [ ] Only public company research goes to external APIs (web search)
- [ ] Clear data flow diagram showing internal vs external processing
- [ ] Audit log of all data movements
- [ ] Configurable data sensitivity tagging
- [ ] Validation that no sensitive fields are sent externally
- [ ] Compliance report generation (GDPR, SOC 2)

#### Priority: MUST HAVE (Regulatory Requirement)
#### Story Points: 13

---

### AS-047: Model Quality Monitoring
**As a** quality assurance lead
**I want to** monitor open-source model output quality
**So that** I can ensure professional-grade results

#### Acceptance Criteria
- [ ] Automated quality scoring for each generated analysis
- [ ] Human review queue for outputs with confidence < 0.85
- [ ] Quality metrics dashboard (hallucination rate, coherence, style)
- [ ] Comparison to baseline (Claude API output for validation)
- [ ] Feedback collection from users (thumbs up/down)
- [ ] Quality trend tracking over time
- [ ] Alerts when quality drops below threshold

#### Priority: MUST HAVE
#### Story Points: 8

---

### AS-048: Fine-Tuning Pipeline
**As a** ML engineer
**I want to** fine-tune open-source models on IER corpus
**So that** output quality matches Big 4 professional standards

#### Acceptance Criteria
- [ ] Data collection: 100-150 IER section → Analysis pairs
- [ ] Training data preparation and validation
- [ ] Fine-tuning workflow (Qwen 72B on IER writing style)
- [ ] Evaluation metrics (BLEU score, human eval, style consistency)
- [ ] A/B testing: base model vs fine-tuned model
- [ ] Model versioning and rollback capability
- [ ] Periodic retraining with new examples (monthly)

#### Priority: SHOULD HAVE (Phase 2 - Months 4-6)
#### Story Points: 21

---

### AS-049: Hybrid Model Fallback
**As a** system architect
**I want to** use Claude API for edge cases
**So that** quality is maintained for complex analyses

#### Acceptance Criteria
- [ ] Automatic detection of complex queries (ambiguous, multi-domain)
- [ ] Fallback to Claude API for quality polish (configurable threshold)
- [ ] Data sanitization before Claude API call (remove sensitive data)
- [ ] Cost tracking for hybrid usage
- [ ] User notification when Claude API is used
- [ ] Configuration: primary model (Qwen), fallback model (Claude), threshold
- [ ] Gradual reduction of fallback usage as fine-tuning improves

#### Priority: SHOULD HAVE (Months 3-6)
#### Story Points: 8

---

### AS-050: Context Window Optimization
**As a** system engineer
**I want to** optimize content to fit within open-source model context windows
**So that** we can process maximum relevant information

#### Acceptance Criteria
- [ ] Dynamic context pruning when approaching limit (128K tokens)
- [ ] Prioritization: Most recent IER (full) + older IERs (summarized)
- [ ] Token counting before LLM call (avoid overflow errors)
- [ ] Intelligent section pruning (keep key subsections only)
- [ ] Sliding window approach for very long content
- [ ] Context usage metrics (average tokens used, peak usage)
- [ ] Warnings when content must be truncated

#### Priority: MUST HAVE
#### Story Points: 8

---

## Epic 7: Performance & Reliability

### AS-016: Fast Processing of IER Content
**As a** time-constrained user  
**I want** rapid analysis results  
**So that** I can meet tight deadlines

#### Acceptance Criteria
- [ ] Initial results within 30 seconds
- [ ] Full analysis within 2 minutes
- [ ] Progress indicator during processing
- [ ] Can queue multiple analyses
- [ ] Background processing option
- [ ] Email notification when complete

#### Dependencies
- AS-022 (LLM Processing Pipeline)

#### Priority: MUST HAVE
#### Story Points: 8

---

### AS-029: Graceful Degradation
**As a** user
**I want** the system to provide partial results when some IERs fail
**So that** I still get value even if processing isn't perfect

#### Acceptance Criteria
- [ ] Analysis succeeds with 30%+ IER success rate
- [ ] Clear indication of partial results (warning banner)
- [ ] Shows which IERs failed and why
- [ ] Adjusted confidence scores for partial results
- [ ] Option to retry failed IERs
- [ ] Logs failures for system improvement

#### Dependencies
- AS-004 (Extract Industry Overview)

#### Priority: MUST HAVE
#### Story Points: 5

---

### AS-017: Handle Large Batch Analysis
**As a** research team  
**I want to** analyze multiple targets at once  
**So that** I can screen opportunities efficiently

#### Acceptance Criteria
- [ ] Can input up to 10 targets at once
- [ ] Parallel processing of requests
- [ ] Batch report generation
- [ ] Progress tracking dashboard
- [ ] Partial results available as completed
- [ ] Export batch results to CSV

#### Priority: NICE TO HAVE
#### Story Points: 13

---

### AS-018: Confidence Scoring
**As a** decision maker  
**I want to** know how confident the system is  
**So that** I can assess reliability

#### Acceptance Criteria
- [ ] Shows confidence score for each insight
- [ ] Explains factors affecting confidence
- [ ] Higher confidence with more IER matches
- [ ] Lower confidence with limited data
- [ ] Confidence breakdown by section
- [ ] Overall analysis confidence rating

#### Dependencies
- AS-005 (Aggregate Industry Insights)

#### Priority: SHOULD HAVE
#### Story Points: 8

---

### AS-030: Cost Tracking Dashboard
**As an** admin
**I want to** monitor LLM API costs in real-time
**So that** I can control spending and optimize usage

#### Acceptance Criteria
- [ ] Real-time cost dashboard (current month)
- [ ] Cost per analysis breakdown
- [ ] Alerts when approaching budget limits
- [ ] Cache hit rate monitoring
- [ ] Token usage trends over time
- [ ] Cost optimization recommendations

#### Dependencies
- AS-022 (LLM Processing Pipeline)

#### Priority: SHOULD HAVE
#### Story Points: 5

---

## Epic 8: End-User Intelligence Features

### AS-031: Industry Trend Alerts
**As an** M&A professional
**I want to** be notified of significant industry changes
**So that** I can spot opportunities early

#### Acceptance Criteria
- [ ] Set up alerts for specific industries
- [ ] Detect: new IERs, changing multiples, opinion shifts
- [ ] Email/SMS notification options
- [ ] Alert frequency settings (real-time, daily, weekly)
- [ ] Customizable alert thresholds
- [ ] Alert history and management

#### Dependencies
- AS-002 (Find Relevant IERs)

#### Priority: NICE TO HAVE
#### Story Points: 8

---

### AS-032: Competitive Intelligence
**As a** corporate development executive
**I want to** track my competitors' M&A activity
**So that** I can understand their strategy and respond

#### Acceptance Criteria
- [ ] Watchlist of competitor companies
- [ ] Automatic detection when competitors appear in IERs
- [ ] Competitor acquisition pattern analysis
- [ ] Geographic expansion trends
- [ ] Valuation multiples paid by competitors
- [ ] Downloadable competitor activity report

#### Dependencies
- AS-002 (Find Relevant IERs)

#### Priority: NICE TO HAVE
#### Story Points: 13

---

### AS-033: Market Heat Map
**As a** strategy consultant
**I want to** visualize M&A activity across industries and geographies
**So that** I can identify hot and cold markets

#### Acceptance Criteria
- [ ] Interactive geographic heat map
- [ ] Color intensity by deal volume/value
- [ ] Filter by time period, industry, deal size
- [ ] Click region to see deal details
- [ ] Trend arrows (heating up/cooling down)
- [ ] Export heat map as image

#### Dependencies
- AS-002 (Find Relevant IERs)

#### Priority: NICE TO HAVE
#### Story Points: 13

---

### AS-034: Expert Firm League Table
**As a** corporate manager selecting advisors
**I want to** see rankings of expert firms
**So that** I can choose the right advisor for my transaction

#### Acceptance Criteria
- [ ] League table by number of opinions written
- [ ] Filter by industry, geography, deal size
- [ ] Average opinion stance (% fair vs not fair)
- [ ] Typical methodologies used
- [ ] Client list and notable deals
- [ ] Download league table as PDF

#### Dependencies
- AS-007 (Extract Valuation Summary)

#### Priority: SHOULD HAVE
#### Story Points: 8

---

### AS-035: "What-If" Scenario Analysis
**As an** investment banker
**I want to** adjust assumptions and see impact on valuation
**So that** I can prepare for client negotiations

#### Acceptance Criteria
- [ ] Adjust: multiple ranges, premiums, growth rates
- [ ] Real-time recalculation of valuations
- [ ] Side-by-side scenario comparison (base vs optimistic vs conservative)
- [ ] Visual sensitivity charts
- [ ] Save scenarios for later reference
- [ ] Include scenarios in exported reports

#### Dependencies
- AS-009 (Multiple Benchmarking)

#### Priority: SHOULD HAVE
#### Story Points: 13

---

### AS-036: Plain English Insights
**As a** non-technical executive
**I want** analysis explained in simple business language
**So that** I can understand without financial jargon

#### Acceptance Criteria
- [ ] Toggle between "Technical" and "Executive" views
- [ ] Replaces jargon (e.g., "EV/EBITDA" → "company value vs profit")
- [ ] One-paragraph executive summary at top
- [ ] Visual indicators (👍👎) for good/bad signals
- [ ] Highlights key takeaways in bullet points
- [ ] "Explain this" button for complex terms

#### Dependencies
- AS-010 (Generate Buy-Side Report)

#### Priority: SHOULD HAVE
#### Story Points: 8

---

## Epic 9: Integration & API

### AS-019: API Access for Analysis
**As a** system integrator  
**I want** API access to analysis functions  
**So that** I can integrate with other tools

#### Acceptance Criteria
- [ ] RESTful API endpoints
- [ ] Authentication via API keys
- [ ] Rate limiting (100 requests/hour)
- [ ] Webhook support for async processing
- [ ] Comprehensive API documentation
- [ ] SDKs for Python/JavaScript

#### Dependencies
- None (integration story)

#### Priority: NICE TO HAVE
#### Story Points: 13

---

### AS-020: Connect to Transaction Insight
**As a** system user  
**I want** seamless connection to Transaction Insight  
**So that** I have access to all extracted IER data

#### Acceptance Criteria
- [ ] Real-time sync with Transaction Insight
- [ ] Access to all IER metadata
- [ ] Can retrieve IER PDFs on demand
- [ ] Inherits user permissions
- [ ] Handles connection failures gracefully
- [ ] Caches frequently accessed data

#### Dependencies
- None (integration story)

#### Priority: MUST HAVE
#### Story Points: 8

---

## Story Prioritization Matrix

### Phase 1: MVP with Open-Source Models (Months 1-3)

#### Sprint 0 (Weeks 1-2): Infrastructure Foundation
- AS-045: Self-Hosted Model Deployment (21 pts) - **CRITICAL PATH**
- AS-046: Sensitive Data Isolation (13 pts) - **CRITICAL PATH**
- AS-020: Connect to Transaction Insight (8 pts)
Total: 42 points (2-week sprint, critical infrastructure)

#### Sprint 1 (Weeks 3-4): Core Extraction Pipeline
- AS-021: Smart Section Detection (8 pts)
- AS-022: LLM Processing Pipeline (13 pts)
- AS-050: Context Window Optimization (8 pts)
- AS-026: Real-Time Progress Indicator (5 pts)
Total: 34 points

#### Sprint 2 (Weeks 5-6): IER Discovery & Industry Analysis
- AS-001: Input Target Details (5 pts)
- AS-002: Find Relevant IERs (8 pts)
- AS-003: Handle Unknown Size (5 pts)
- AS-004: Extract Industry Overview (13 pts)
Total: 31 points

#### Sprint 3 (Weeks 7-8): Valuation - Comparable Companies
- AS-007: Extract Expert-Curated Comparables (13 pts) - **NEW CORE VALUE**
- AS-007b: Refresh with Live Market Data (8 pts) - **NEW CORE VALUE**
- AS-007c: Extract Precedent Transactions (8 pts)
Total: 29 points

#### Sprint 4 (Weeks 9-10): Analysis & Reporting
- AS-005: Aggregate Industry Insights (8 pts)
- AS-009: Multiple Benchmarking (8 pts)
- AS-047: Model Quality Monitoring (8 pts) - **QUALITY ASSURANCE**
- AS-024: Deal Similarity Scoring (5 pts)
Total: 29 points

#### Sprint 5 (Weeks 11-12): Report Generation & MVP Polish
- AS-010: Generate Buy-Side Report (13 pts)
- AS-012: Export to Excel (5 pts)
- AS-018: Confidence Scoring (8 pts)
- AS-029: Graceful Degradation (5 pts)
Total: 31 points

**Phase 1 Complete**: MVP with 75-80% quality, data privacy, core valuation features

---

### Phase 2: Quality Enhancement (Months 4-6)

#### Sprint 6 (Weeks 13-14): Fine-Tuning Preparation
- AS-048: Fine-Tuning Pipeline (21 pts) - **QUALITY IMPROVEMENT**
- AS-049: Hybrid Model Fallback (8 pts)
Total: 29 points

#### Sprint 7 (Weeks 15-16): User Experience Enhancement
- AS-016: Fast Processing (8 pts)
- AS-025: Save Analysis as Template (5 pts)
- AS-013a: Basic Question Answering (8 pts)
- AS-036: Plain English Insights (8 pts)
Total: 29 points

#### Sprint 8 (Weeks 17-18): Advanced Features
- AS-011: Sell-Side Report (13 pts)
- AS-008: Football Field Chart (13 pts)
Total: 26 points

#### Sprint 9 (Weeks 19-20): Cost & Performance Optimization
- AS-030: Cost Tracking Dashboard (5 pts)
- AS-006: Risk Analysis (8 pts)
- AS-015: Track History (5 pts)
- AS-034: Expert Firm League Table (8 pts)
Total: 26 points

**Phase 2 Complete**: 85-90% quality, fine-tuned model, enhanced UX

---

### Phase 3: RAG & Scale (Months 7-12)

#### Sprint 10+: RAG Layer (Future)
- Vector database implementation for broker research
- Hybrid retrieval (metadata + vector search)
- Conversational follow-ups
- Cross-domain queries
- Dataset expansion (10,000+ IERs, broker research, news)

### Backlog (Future Sprints)

**High Priority Backlog** (Next 2-3 sprints):
- AS-011: Sell-Side Report (13 pts)
- AS-008: Football Field Chart (13 pts)
- AS-006: Risk Analysis (8 pts)
- AS-015: Track History (5 pts)
- AS-030: Cost Tracking Dashboard (5 pts)
- AS-035: What-If Scenario Analysis (13 pts)

**Medium Priority Backlog**:
- AS-014: Compare IERs (8 pts)
- AS-013b: Follow-Up Questions (8 pts)
- AS-013c: Learning from Feedback (5 pts)
- AS-017: Batch Analysis (13 pts)
- AS-027: Bookmark and Annotate (5 pts)
- AS-028: Analysis Comparison (8 pts)

**Nice to Have Backlog**:
- AS-019: API Access (13 pts)
- AS-031: Industry Trend Alerts (8 pts)
- AS-032: Competitive Intelligence (13 pts)
- AS-033: Market Heat Map (13 pts)

---

## Definition of Ready
Before starting any story:
- [ ] Acceptance criteria defined
- [ ] Dependencies identified
- [ ] Technical approach agreed
- [ ] Test scenarios documented
- [ ] UI mockups created (if applicable)

## Definition of Done
Story is complete when:
- [ ] All acceptance criteria met
- [ ] Unit tests written (>80% coverage)
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Code reviewed and approved
- [ ] Performance benchmarks met
- [ ] Deployed to staging environment

---

## Technical Considerations

### Performance Requirements
- Industry Overview extraction: < 30 seconds
- Full analysis: < 2 minutes
- Report generation: < 10 seconds
- API response time: < 2 seconds

### Scalability Requirements
- Support 100+ concurrent users
- Process 50+ IERs simultaneously
- Handle IERs up to 200 pages
- Store 10,000+ analysis results

### Data Requirements
- Access to Transaction Insight API
- IER PDF storage (S3)
- Analysis results database
- User session management
- Cache layer (Redis)

---

## Story Dependencies Graph

```
Foundation Layer (Sprint 0):
AS-021 (Section Detection) ──┐
AS-022 (LLM Pipeline) ────────┼──→ AS-004 (Industry Overview)
AS-020 (Transaction Insight)─┘         │
                                       ├──→ AS-005 (Aggregate Insights)
AS-001 (Input) ──→ AS-002 (Find IERs)─┤         │
                            │          └──→ AS-007 (Valuation) ──→ AS-009 (Benchmarking)
                            │                      │                     │
                            └──→ AS-003 (Unknown Size)                  │
                                        │                                 │
                                        └──→ AS-023 (Quality) ───────────┤
                                                                          │
                                                                          ├──→ AS-010 (Buy Report)
                                                                          │         │
                                                                          │         └──→ AS-012 (Excel)
                                                                          │
                                                                          └──→ AS-011 (Sell Report)
```

## End-User Perspective: Additional Ideas

### Discovered User Needs (Based on typical M&A workflows)

1. **AS-037: Quick Sanity Check Mode** (5 pts)
   - "Is this valuation reasonable?" - Yes/No + 3 bullet points
   - For busy executives who need quick validation
   - < 10 second response time
   - Mobile-friendly interface

2. **AS-038: Red Flag Detector** (8 pts)
   - Automatically highlights concerning patterns
   - "87% of similar deals had negative opinions"
   - "This multiple is in top 5% - justify carefully"
   - Warning badges on suspicious data points

3. **AS-039: Presentation Mode** (5 pts)
   - Clean, client-ready output (no internal jargon)
   - One-click copy charts to PowerPoint
   - Hide data quality warnings/caveats for external sharing
   - Branding customization (logo, colors)

4. **AS-040: Email Digest** (5 pts)
   - Weekly summary of new IERs in watched industries
   - "3 new medical device deals this week"
   - Key metrics summary (avg multiples, opinions)
   - One-click to run full analysis

5. **AS-041: Mobile Quick View** (8 pts)
   - Mobile-responsive interface
   - Key metrics dashboard
   - Push notifications for analysis completion
   - Voice input for target details

6. **AS-042: Collaboration Features** (8 pts)
   - Share analysis with colleagues (view-only link)
   - Comments on specific sections
   - @mention team members
   - Version history (who changed what)

7. **AS-043: Learning Center** (3 pts)
   - "What is EV/EBITDA?" - contextual help
   - Video tutorials for first-time users
   - Example analyses to explore
   - Tips based on usage patterns

8. **AS-044: Data Room Integration** (13 pts)
   - Upload target company's data room
   - Auto-extract financial data
   - Compare target vs benchmark IERs
   - Gap analysis (what data is missing)

---

## Summary of Key Updates

### New Core Value Stories
- **AS-007, AS-007b, AS-007c**: Expert-curated comparable companies extraction with live data refresh (core valuation advantage)
- **AS-045**: Self-hosted model deployment for data privacy
- **AS-046**: Sensitive data isolation (regulatory compliance)
- **AS-047**: Model quality monitoring
- **AS-048**: Fine-tuning pipeline (quality improvement roadmap)
- **AS-049**: Hybrid model fallback strategy
- **AS-050**: Context window optimization for open-source models

### Architecture Alignment
- **Phase 1 (Months 1-3)**: MVP with open-source models, 75-80% quality, data privacy
- **Phase 2 (Months 4-6)**: Fine-tuning for 85-90% quality, enhanced UX
- **Phase 3 (Months 7-12)**: RAG layer for dataset expansion

### Critical Path
1. Self-hosted model infrastructure (Sprint 0)
2. Section extraction pipeline (Sprints 1-2)
3. Comparable companies extraction (Sprint 3) - **Core differentiation**
4. Analysis and reporting (Sprints 4-5)
5. Fine-tuning and quality improvement (Sprints 6+)

---

*Document Version: 3.0*
*Created: 30/09/2025*
*Last Updated: 30/09/2025*
*Major Update: Added data privacy stories, open-source model management, comparable company extraction focus, phased sprint planning aligned with architecture decisions*