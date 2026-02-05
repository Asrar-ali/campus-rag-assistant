from app.chunking import chunk_pages

def test_chunk_pages_basic():
    pages = ["a" * 5000]
    pnums = [1]
    chunks = chunk_pages(pages, pnums, chunk_chars=3000, overlap_chars=400, max_chunks=100)
    assert len(chunks) >= 2
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
