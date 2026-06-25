# docs/ — screenshots & demo assets

Drop recruiter-facing visuals here. The main README references these filenames, so
keeping the names exact means you only have to uncomment the image lines (no path edits).

## Expected files

| File | What it should show | Referenced in |
|---|---|---|
| `screenshot.png` | The Streamlit app with a question answered (context visible) | README → Screenshots |
| `demo.gif` | A 15–30s loop: upload a PDF → ask a question → streamed answer | README → Screenshots |

## How to capture

1. Run the app locally (`streamlit run streamlit_app.py`) or open the live Cloud URL.
2. Upload a sample PDF and ask a grounded question; wait for the streamed answer.
3. **Screenshot:** capture the full app window → save as `docs/screenshot.png`.
4. **GIF:** record the upload→ask→stream flow (e.g. ScreenToGif, Kap, or Peek) →
   export as `docs/demo.gif` (keep it under ~5 MB so it loads fast on GitHub).

## Then

Uncomment the image lines in the root [`README.md`](../README.md) Screenshots section.
A real screenshot is the highest-impact addition for recruiter review.
