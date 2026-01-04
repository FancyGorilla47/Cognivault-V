import os
import urllib.parse
import re
import io
import json
import fitz # PyMuPDF
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

# Load environment variables
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
# Azure Search
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "universal-media-index-v1"

# Azure OpenAI
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("OPENAI_KEY")
EMBEDDING_DEPLOYMENT = "text-embedding-3-large"
# Using the same deployment for chat as in reference, or default to gpt-4o/gpt-4
CHAT_DEPLOYMENT = "gpt-4.1-795005" 

# Azure Blob Storage
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
CONTAINER_NAME = "jan5demo"

# Initialize Clients
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_KEY)
)

openai_client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version="2023-05-15",
    azure_endpoint=OPENAI_ENDPOINT
)

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==========================================
# HELPERS
# ==========================================
def get_media_sas_url(blob_name: str, media_type: str = "video") -> str:
    """Generates a read-only SAS URL for a blob, handling subfolder prefixes."""
    try:
        # Determine Folder Prefix
        # Note: Index stores just 'filename.ext', strictly.
        # But blobs are in subfolders: 'videos/', 'images/', 'pdfs/'.
        prefix = ""
        m_type = media_type.lower()
        if "video" in m_type:
            prefix = "videos/"
        elif "image" in m_type:
            prefix = "images/"
        elif "pdf" in m_type or "doc" in m_type:
            prefix = "pdfs/"
            
        full_blob_path = f"{prefix}{blob_name}"

        # Parse connection string for account key/name
        conn_str_parts = {
            item.split('=', 1)[0]: item.split('=', 1)[1] 
            for item in BLOB_CONNECTION_STRING.split(';') if '=' in item
        }
        account_name = conn_str_parts.get("AccountName")
        account_key = conn_str_parts.get("AccountKey")

        if not account_name or not account_key:
            return f"https://{container_client.account_name}.blob.core.windows.net/{CONTAINER_NAME}/{full_blob_path}"

        expiry_time = datetime.now(timezone.utc) + timedelta(hours=2)
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=CONTAINER_NAME,
            blob_name=full_blob_path,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry_time
        )
        
        # Proper encoding for URL
        encoded_name = urllib.parse.quote(full_blob_path)
        return f"https://{account_name}.blob.core.windows.net/{CONTAINER_NAME}/{encoded_name}?{sas_token}"
    except Exception as e:
        print(f"Error generating SAS: {e}")
        return ""

def generate_embedding(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding

# ==========================================
# API MODELS
# ==========================================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    video_url: Optional[str] = None
    start_time_seconds: Optional[int] = None
    debug_context: Optional[str] = None
    auto_open: Optional[dict] = None

# ==========================================
# ROUTES
# ==========================================

@app.get("/")
def read_root():
    # Serve the frontend
    if os.path.exists("chat_interface.html"):
        return FileResponse("chat_interface.html")
    return {"error": "chat_interface.html not found"}

@app.get("/view_pdf")
def view_pdf(blob_name: str, chunk_source: Optional[str] = None):
    """
    Fetches PDF, applies server-side highlighting to the specific chunks, and returns the stream.
    """
    try:
        # 1. Get Blob Client
        blob_client = container_client.get_blob_client(blob_name)
        if not blob_client.exists():
            raise HTTPException(status_code=404, detail="File not found: " + blob_name)

        # 2. Download Content
        stream_bytes = blob_client.download_blob().readall()
        doc = fitz.open(stream=stream_bytes, filetype="pdf")

        # 3. Apply Highlights
        if chunk_source:
             # Split segments: D(page, ...); D(page, ...)
             segments = chunk_source.split(';')
             for seg in segments:
                 seg = seg.strip()
                 if not seg.startswith("D("): continue
                 
                 # Extract numbers
                 nums = re.findall(r"[\d\.]+", seg)
                 if not nums or len(nums) < 9: continue
                 
                 # Parse
                 vals = list(map(float, nums))
                 page_num = int(vals[0]) - 1 # 1-based to 0-based
                 
                 # Ensure page exists
                 if 0 <= page_num < len(doc):
                     page = doc[page_num]
                     
                     # Polygon coords (Inches) -> [x1, y1, x2, y2, x3, y3, x4, y4]
                     # PyMuPDF expects Points (1 inch = 72 points)
                     poly_inches = vals[1:]
                     if len(poly_inches) >= 8:
                         points = [p * 72 for p in poly_inches]
                         
                         # Create Quad (TL, TR, BL, BR) -> PyMuPDF Quad(ul, ur, ll, lr)
                         # Azure: TL(0,1), TR(2,3), BR(4,5), BL(6,7)
                         # Quad expects: UL, UR, LL, LR
                         
                         quad = fitz.Quad(
                            fitz.Point(points[0], points[1]), # TL
                            fitz.Point(points[2], points[3]), # TR
                            fitz.Point(points[6], points[7]), # BL
                            fitz.Point(points[4], points[5])  # BR
                         )
                         
                         # Add highlight
                         annot = page.add_highlight_annot(quad)
                         annot.set_colors(stroke=[1, 1, 0]) # Yellow
                         annot = page.add_highlight_annot(quad)
                         annot.set_colors(stroke=[1, 1, 0]) # Yellow
                         annot.update()
        
        # 4. Set Open Action to First Highlighted Page
        # Find the first page mentioned in chunk_source to jump to
        start_page_idx = 0
        target_override = None # (page_xref, left, top)
        
        if chunk_source:
             match = re.search(r"D\((\d+),", chunk_source)
             if match:
                 p_num = int(match.group(1)) - 1
                 if 0 <= p_num < len(doc):
                     start_page_idx = p_num
                     
                     # Try to get coordinates for refined scrolling (/XYZ)
                     try:
                        # Extract first polygon
                        # D(1, x1, y1, x2, y2, x3, y3, x4, y4)
                        # We need x1 (left) and y1 (top) - wait, y1 in PyMuPDF is top-down.
                        # We need to convert to PDF Native (Bottom-Up) for OpenAction '/XYZ'
                        
                        # Re-parse segments to find the first one for this page
                        segments = chunk_source.split(';')
                        first_seg = segments[0]
                        nums = re.findall(r"[\d\.]+", first_seg)
                        if nums and len(nums) >= 9:
                             vals = list(map(float, nums))
                             # vals[1] = x1 (inches), vals[2] = y1 (inches)
                             
                             # Convert to Points (1 inch = 72 pts)
                             x_inch = vals[1]
                             y_inch = vals[2] 
                             
                             x_pt = x_inch * 72
                             y_pt = y_inch * 72
                             
                             current_page = doc[start_page_idx]
                             page_height = current_page.rect.height
                             
                             # PyMuPDF y is from top. PDF Native y is from bottom.
                             # If Azure returns coordinates where (0,0) is top-left (standard for OCR),
                             # Then y_pt is distance from top.
                             # PDF Native Top = PageHeight - y_pt.
                             # We want to scroll so this point is near the top of the view.
                             # /XYZ left top zoom
                             
                             # Let's add a small margin (e.g. 50 pts) so it's not right at the edge
                             target_left = x_pt
                             target_top = page_height - y_pt + 20 # slightly above the highlight
                             
                             target_override = (current_page.xref, target_left, target_top)
                     except Exception as ex:
                         print(f"Coord calc failed: {ex}")

        try:
            if target_override:
                # /XYZ left top zoom (0 = retain zoom)
                cmd = f"[{target_override[0]} 0 R /XYZ {target_override[1]} {target_override[2]} 0]"
                doc.xref_set_key(doc.pdf_catalog, "OpenAction", cmd)
            else:
                # Fallback to page fit
                page_xref = doc[start_page_idx].xref
                cmd = f"[{page_xref} 0 R /Fit]"
                doc.xref_set_key(doc.pdf_catalog, "OpenAction", cmd)
        except Exception as e:
            print(f"Failed to set OpenAction: {e}")

        # Set PDF Metadata for Viewer Title
        clean_title = os.path.basename(blob_name)
        doc.set_metadata({"title": clean_title})

        # 4. Return Modified PDF
        output_stream = io.BytesIO(doc.tobytes())
        return StreamingResponse(
            output_stream, 
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{clean_title}"'}
        )

    except Exception as e:
        print(f"View PDF Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    user_query = request.message
    print(f"Received query: {user_query}")

    # 1. Embed Query
    query_vector = generate_embedding(user_query)

    # 2. Search Index (Hybrid Search: Vector + Keyword)
    # Using 'universal-media-index-v1' fields: content_summary, file_id, start_time_ms, file_name, media_type
    # We now pass 'search_text=user_query' to enable BM25 keyword search on text fields (like visible_text)
    # This matches the user's need to find images via OCR text.
    results = search_client.search(
        search_text=user_query,
        vector_queries=[
            VectorizedQuery(vector=query_vector, k_nearest_neighbors=10, fields="content_vector")
        ],
        select=[
            "content_summary", "file_id", "file_name", "start_time_ms", "media_type", "page_number", "chunk_source",
            "deduced_context", "visual_summary", "speech_summary", "sentiment", "visible_objects", "visible_text"
        ],
        top=10
    )

    top_results = list(results)
    if not top_results:
        print("No results found in search index.")
        return ChatResponse(answer="I couldn't find any relevant information in the video library.")

    # LOGGING
    print(f"\n--- AI Search Results ({len(top_results)}) ---")
    for idx, doc in enumerate(top_results):
        score = doc.get("@search.score", "N/A")
        title = doc.get("file_name", "No Title")
        content_snippet = doc.get("content_summary", "").replace("\n", " ")[:100]
        print(f"[{idx+1}] Score: {score} | File: {title}")
        print(f"    Snippet: {content_snippet}...")
    print("-------------------------------------------\n")

    # 3. Construct Context for LLM
    context_map = {}
    context_str = ""

    for i, doc in enumerate(top_results):
        ref_id = f"SEGMENT_{i+1}"
        
        # Store actual data needed for "onClick"
        context_map[ref_id] = {
            "start_time_ms": doc.get("start_time_ms", 0),
            "file_id": doc.get("file_id"),
            "file_name": doc.get("file_name", "Unknown File"),
            "media_type": doc.get("media_type", "video"),
            "page_number": doc.get("page_number"),
            "chunk_source": doc.get("chunk_source")
        }

        # Build prompt context with rich fields
        context_str += f"\n[[{ref_id}]]\nSource: {doc.get('file_name')}\n"
        
        # Add available fields if they exist
        if doc.get('content_summary'):
            context_str += f"Content: {doc.get('content_summary')}\n"
        
        if doc.get('deduced_context'):
            context_str += f"Deduced Context: {doc.get('deduced_context')}\n"
            
        if doc.get('visual_summary'):
            context_str += f"Visual Summary: {doc.get('visual_summary')}\n"
            
        if doc.get('speech_summary'):
            context_str += f"Speech Summary: {doc.get('speech_summary')}\n"
            
        if doc.get('sentiment'):
            context_str += f"Sentiment: {doc.get('sentiment')}\n"
            
        if doc.get('visible_objects'):
            # simple join if it's a list
            objs = doc.get('visible_objects')
            if isinstance(objs, list):
                context_str += f"Visible Objects: {', '.join(str(x) for x in objs)}\n"
            else:
                context_str += f"Visible Objects: {objs}\n"
                
        if doc.get('visible_text'):
            # simple join if it's a list
            vtext = doc.get('visible_text')
            if isinstance(vtext, list):
                 context_str += f"Visible Text: {', '.join(str(x) for x in vtext)}\n"
            else:
                 context_str += f"Visible Text: {vtext}\n"

    # 4. Generate Answer with LLM
    system_prompt = (
        "You are a helpful CogniVault AI assistant. Your goal is to answer questions using strictly the provided context. \n\n"
        "**CRITICAL RULES:**\n"
        "1. **GREETINGS**: If the user says 'hello', greet them warmly. Do NOT cite context for greetings.\n"
        "2. **OUT OF SCOPE**: If the answer is not in the context, say 'I cannot answer this based on the available videos.'\n"
        "3. **STRICT CONTEXT**: Answer using ONLY the provided segments [[SEGMENT_X]].\n"
        "4. **CITATIONS**: Cite [[SEGMENT_X]] exactly where relevant.\n"
        "5. **FORMATTING**: Use Markdown headers, bolding for steps, and lists.\n"
        "6. **OUTPUT JSON**: Return a valid JSON object: { \"answer\": \"...\" }\n"
        "   Example: { \"answer\": \"To do X, follow these steps:\\n- Step 1 [[SEGMENT_1]]\" }"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_query}"}
    ]

    try:
        completion = openai_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=messages,
            temperature=0.0,
            response_format={ "type": "json_object" }
        )
        response_content = completion.choices[0].message.content
        
        # Parse JSON
        import json
        parsed_response = json.loads(response_content)
        ai_answer = parsed_response.get("answer", "Here is the information you requested.")
        
        # 5. Process References (Inject Buttons & Build Footer)
        ref_counter = 1
        segment_to_number = {}
        first_video_url = None
        first_start_time = None
        
        # Track which segments were actually used in the response
        used_segment_ids = set()

        def replace_match(match):
            nonlocal ref_counter, first_video_url, first_start_time
            full_match_text = match.group(0)
            segments_found = re.findall(r"(SEGMENT_\d+)", full_match_text)
            
            buttons_html = ""
            
            for segment_id in segments_found:
                if segment_id in context_map:
                    used_segment_ids.add(segment_id) # Track usage
                    
                    # Numbering Logic
                    if segment_id in segment_to_number:
                        display_number = segment_to_number[segment_id]
                    else:
                        display_number = ref_counter
                        segment_to_number[segment_id] = ref_counter
                        ref_counter += 1
                        
                    # Create a small inline citation style (optional, or keep as button)
                    # For now, we keep the inline button but also add to footer
                    # The user asked for "refrences at the bottom", but usually inline is good too.
                    # Let's keep inline simple buttons.
                    
                    # We won't generate the full action here if we want strict bottom ref, 
                    # but typically inline [1] is useful.
                    # Let's return just the number [1] or a small clickable superscript.
                    # But to preserve existing behavior requested: "references at the bottom... videos: button".
                    # So inline can just be the number [1].
                    
                    buttons_html += f' <span style="font-weight:bold; color:#0045d0;">[{display_number}]</span>'
            
            return buttons_html if buttons_html else ""

        # Replace [[SEGMENT_X]] tags with simple numbers in text
        processed_answer = re.sub(r"\[\[.*?SEGMENT_\d+.*?\]\]", replace_match, ai_answer)
        
        # --- BUILD FOOTER ---
        # Group used segments by media type
        grouped_refs = {"video": [], "image": [], "pdf": []}
        
        # Sort used segments by their assigned number
        sorted_used_segments = sorted(list(used_segment_ids), key=lambda x: segment_to_number.get(x, 999))
        
        auto_open_data = None
        
        for seg_id in sorted_used_segments:
            data = context_map[seg_id]
            media_type = data.get("media_type", "video").lower()
            file_name = data.get("file_name", "Unknown File")
            display_num = segment_to_number[seg_id]
            
            # Prepare data for button
            ref_data = {
                "num": display_num,
                "name": file_name,
                "url": "",
                "onclick": ""
            }
            
            # Generate URL based on type
            # Pass media_type to get correct folder prefix
            sas_url = get_media_sas_url(file_name, media_type) 
            
            current_media_info = None

            if "video" in media_type:
                start_time = int(data.get("start_time_ms", 0) / 1000)
                ref_data["onclick"] = f"playRefVideo('{sas_url}', {start_time})"
                grouped_refs["video"].append(ref_data)
                
                # Capture for potential auto-open
                current_media_info = {
                    "type": "video",
                    "url": sas_url,
                    "start_time": start_time
                }
                
            elif "image" in media_type:
                safe_title = file_name.replace("'", "\\'")
                ref_data["onclick"] = f"showImage('{sas_url}', '{safe_title}')"
                grouped_refs["image"].append(ref_data)
                
                current_media_info = {
                    "type": "image",
                    "url": sas_url,
                    "title": file_name
                }
                
            elif "pdf" in media_type:
                page = data.get("page_number") 
                chunk_source = data.get("chunk_source", "")
                
                safe_page = page if page else "1"
                safe_coords = chunk_source.replace("'", "\\'") if chunk_source else ""
                
                ref_data["onclick"] = f"showPdf('{sas_url}', {safe_page}, '{safe_coords}')"
                grouped_refs["pdf"].append(ref_data)
                
                current_media_info = {
                    "type": "pdf",
                    "url": sas_url,
                    "page": safe_page,
                    "chunk_source": chunk_source
                }
            
            # Use FIRST media encountered for auto-open (if not set yet)
            if auto_open_data is None and current_media_info:
                auto_open_data = current_media_info

        # Construct HTML Footer
        footer_html = '<div class="sources-footer">'
        has_refs = False
        
        # Videos
        if grouped_refs["video"]:
            has_refs = True
            footer_html += '<div class="source-group"><strong>Videos:</strong>'
            for ref in grouped_refs["video"]:
                footer_html += f'<button class="ref-btn" onclick="{ref["onclick"]}">{ref["num"]}</button>'
            footer_html += '</div>'
            
        # Images
        if grouped_refs["image"]:
            has_refs = True
            footer_html += '<div class="source-group"><strong>Images:</strong>'
            for ref in grouped_refs["image"]:
                footer_html += f'<button class="ref-btn" onclick="{ref["onclick"]}">{ref["num"]}</button>'
            footer_html += '</div>'
            
        # PDFs
        if grouped_refs["pdf"]:
            has_refs = True
            footer_html += '<div class="source-group"><strong>PDFs:</strong>'
            for ref in grouped_refs["pdf"]:
                footer_html += f'<button class="ref-btn" onclick="{ref["onclick"]}">{ref["num"]}</button>'
            footer_html += '</div>'
            
        footer_html += '</div>'

        if has_refs:
            processed_answer += footer_html

        ai_answer = processed_answer
        
        # variables for return are already set: ai_answer, auto_open_data
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        ai_answer = "I encountered an error generating the response."
        auto_open_data = None

    return ChatResponse(
        answer=ai_answer,
        auto_open=auto_open_data,
        debug_context=context_str[:200] + "..." if 'context_str' in locals() else None
    )

if __name__ == "__main__":
    import uvicorn
    # Standard boilerplate
    uvicorn.run(app, host="0.0.0.0", port=8000)
