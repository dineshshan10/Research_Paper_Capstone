-- Creates a concise, presentation-first PowerPoint deck with visual flows.
tell application "Keynote"
    activate
    set deck to make new document with properties {document theme:theme "White"}
    tell deck
        set slideTexts to {{"Research Paper Answer Bot", "Ask  →  Search  →  Verify  →  Cite\n\nEvidence-first answers across landmark Generative AI papers\nAnalytics Vidhya · GenAI Pinnacle Capstone"}, {"Why this project?", "Research papers are difficult to search, compare, and trust at speed.\n\nThe goal: answer questions from primary sources — with the exact paper and page visible."}, {"Scope at a glance", "5 landmark papers\n4 embedding candidates\n5 retrieval strategies\n3 ways to use the product"}, {"High-level overview", "RESEARCHER QUESTION\n↓\nSEARCH THE FIVE-PAPER CORPUS\n↓\nRELEVANT EVIDENCE?\n↓\nCITED ANSWER  →  FOLLOW-UP QUESTION"}, {"Technical design", "PDFs\n↓\nPage-aware chunks\n↓\nEmbeddings + BM25\n↓\nRetrieve + relevance check\n↓\nGrounded answer + source cards"}, {"When the corpus is not enough", "In-corpus question\n→ grounded answer with citations\n\nOut-of-corpus question\n→ Tavily web search when configured\n→ transparent refusal when offline"}, {"Evaluation", "Compare embeddings\nMiniLM · BGE-base · GTE-large · OpenAI\n\nCompare retrieval\nDense · BM25 · Hybrid · Rerank · MMR\n\nScore with Hit@5 · MRR · nDCG · latency"}, {"Product experience", "Streamlit workspace\nResearch chat · Corpus browser · Evaluation tab\n\nFastAPI\nCited answers for applications\n\nCLI\nQuick terminal research"}, {"What I learned", "Retrieval quality is measurable — not guesswork.\n\nCitations make RAG answers auditable.\n\nA clear fallback is safer than a confident hallucination.\n\nConversation memory needs explicit thread isolation."}, {"Next steps", "Warm neural models and run the full bake-off\nLock the winning production configuration\nAdd figure and table extraction\nDeploy with managed secrets and authentication"}}
        tell first slide
            set object text of default title item to item 1 of item 1 of slideTexts
            set object text of default body item to item 2 of item 1 of slideTexts
        end tell
        repeat with i from 2 to count of slideTexts
            set s to make new slide with properties {base slide:master slide "Title & Bullets"}
            tell s
                set object text of default title item to item 1 of item i of slideTexts
                set object text of default body item to item 2 of item i of slideTexts
            end tell
        end repeat
        export deck to POSIX file "/Users/dinesh/1Tech/AI Pinnacle Course/Capstone_Projects/Research_Paper_Capstone/docs/Research_Paper_Capstone_Presentation.pptx" as Microsoft PowerPoint
        close
    end tell
end tell
