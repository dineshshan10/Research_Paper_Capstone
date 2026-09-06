-- Creates the final 10-slide Keynote deck and exports it as PowerPoint.
tell application "Keynote"
    activate
    set deck to make new document with properties {document theme:theme "White"}
    tell deck
        set slideTexts to {{"Research Paper Answer Bot", "Evidence-first RAG over five landmark Generative AI papers\nAnalytics Vidhya · GenAI Pinnacle Capstone"}, {"Problem and goal", "Researchers need trustworthy answers across long AI papers.\n\nGoal: cited answers, not an ungrounded chat experience."}, {"Corpus and requirements", "Attention Is All You Need · GPT-4 · InstructGPT · Mistral 7B · Gemini 1.0\n\nCovers vector indexing, open + commercial embeddings, retrieval bake-off, RAG, testing and sources."}, {"Architecture", "PyMuPDF → embedding provider → Chroma + BM25 → hybrid/RRF/rerank → relevance grader → grounded generator → paper/page citations\n\nStreamlit, FastAPI and CLI share one core."}, {"Citation-safe ingestion", "Page-aware chunks retain paper title, page, section and chunk ID.\n\nReference sections are excluded from answer context, making each source traceable."}, {"Embedding bake-off", "MiniLM · BGE-base · GTE-large · OpenAI text-embedding-3-small\n\nMeasures Hit@5, MRR, nDCG@5, index time and query latency. Offline fallback is clearly labelled—not presented as neural results."}, {"Retrieval bake-off", "Dense cosine baseline · BM25 · Hybrid RRF · Hybrid rerank · MMR\n\nHybrid rerank is the production candidate: semantic recall plus technical-term precision."}, {"Corrective and conversational RAG", "Thread-isolated history resolves follow-ups.\n\nThe relevance grader uses Tavily when configured or transparently refuses out-of-corpus questions."}, {"Product demonstration", "Streamlit chat: answers, page citations, source excerpts, provenance, scores and latency.\n\nFastAPI: /health, /api/v1/ask, /api/v1/sources. CLI supports terminal demonstrations."}, {"Quality and next steps", "30-question golden set; automated coverage for ingestion, retrieval, citations, memory and refusal.\n\nNext: warm model cache, run neural/commercial bake-off, lock the winner, add figure/table extraction."}}
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
        save deck in POSIX file "/Users/dinesh/1Tech/AI Pinnacle Course/Capstone_Projects/Research_Paper_Capstone/docs/Research_Paper_Capstone_Presentation.key"
        close
    end tell
end tell
