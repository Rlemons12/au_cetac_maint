from flask import Blueprint, request
from werkzeug.utils import secure_filename
from modules.emtacdb.utlity.main_database.database import create_position, add_powerpoint_to_db, add_document_to_db
from modules.configuration.config import PPT2PDF_PPT_FILES_PROCESS, PPT2PDF_PDF_FILES_PROCESS, DATABASE_URL
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
import os
import re
#import comtypes.client
# import pythoncom  # commented out for cross plate form safe
import logging

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))  # Use scoped_session here
session = Session()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all log levels
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", mode='a'),  # Append mode
        logging.StreamHandler()                    # Log to the console
    ]
)

logger = logging.getLogger(__name__)

def extract_title_from_filename(filename):
    # Remove file extensions from filename
    title = os.path.splitext(filename)[0]
    # Remove any additional extensions (e.g., pptx, jpg, etc.)
    title = re.sub(r'\.\w+', '', title)
    return title

#todo: need to make cross-platform safe
def convert_pptx_to_pdf(pptx_file, pdf_file, *, timeout_sec=300):
    """
    Cross-platform PPTX/PPT -> PDF conversion.

    Windows:
      - Uses PowerPoint COM via `comtypes` (pywin32 `pythoncom`) if
        EMTAC_ENABLE_PPTX_TO_PDF is truthy (default: on).
    Linux/macOS (and fallback on Windows when disabled):
      - Uses LibreOffice `soffice --headless`.

    Returns:
        str | None: Absolute path to `pdf_file` on success, else None.
    """
    def _log(level, msg):
        try:
            # Prefer your app's logger if available
            logger_method = getattr(globals().get("logger", None), level, None)
            if logger_method:
                logger_method(msg)
                return
        except Exception:
            pass
        # Fallback to print
        print(f"[{level.upper()}] {msg}")

    try:
        if not pptx_file or not os.path.exists(pptx_file):
            _log("error", f"PPTX not found: {pptx_file}")
            return None

        in_ext = os.path.splitext(pptx_file)[1].lower()
        if in_ext not in (".pptx", ".ppt"):
            _log("error", f"Unsupported extension for PPT conversion: {in_ext}")
            return None

        pdf_file = os.path.abspath(pdf_file)
        out_dir = os.path.dirname(pdf_file)
        os.makedirs(out_dir, exist_ok=True)

        # ------------------ Windows path (PowerPoint COM) ------------------
        if sys.platform == "win32" and os.getenv("EMTAC_ENABLE_PPTX_TO_PDF", "1").lower() in ("1", "true", "yes"):
            _log("debug", "Windows detected; attempting PowerPoint COM via comtypes")
            pythoncom_mod = None
            powerpoint = None
            presentation = None
            try:
                try:
                    import pythoncom as _pythoncom  # pywin32
                    pythoncom_mod = _pythoncom
                except Exception as imp_err:
                    _log("error", f"'pythoncom' not available (is pywin32 installed?): {imp_err}")
                    return None

                try:
                    import comtypes.client as cc
                except Exception as ct_err:
                    _log("error", f"'comtypes' not available: {ct_err}")
                    return None

                pythoncom_mod.CoInitialize()
                _log("debug", "COM initialized")

                powerpoint = cc.CreateObject("PowerPoint.Application")
                try:
                    # Be quiet and headless
                    try:
                        powerpoint.Visible = 0
                    except Exception:
                        pass
                    try:
                        # 1 = ppAlertsNone
                        powerpoint.DisplayAlerts = 1
                    except Exception:
                        pass

                    # PowerPoint is picky—use absolute path
                    in_abs = os.path.abspath(pptx_file)
                    out_abs = os.path.abspath(pdf_file)

                    # Open and SaveAs (32 = ppSaveAsPDF)
                    presentation = powerpoint.Presentations.Open(in_abs, WithWindow=False)
                    presentation.SaveAs(out_abs, 32)
                    _log("info", f"PPTX converted to PDF via PowerPoint COM: {out_abs}")

                    # Sanity check
                    if os.path.exists(out_abs):
                        return out_abs
                    _log("error", "SaveAs reported success, but output PDF not found")
                    return None

                finally:
                    try:
                        if presentation:
                            presentation.Close()
                    except Exception:
                        pass
                    try:
                        if powerpoint:
                            powerpoint.Quit()
                    except Exception:
                        pass
                    try:
                        pythoncom_mod.CoUninitialize()
                        _log("debug", "COM uninitialized")
                    except Exception:
                        pass

            except Exception as e:
                _log("error", f"Windows/COM conversion failed: {e}")
                # fall through to try LibreOffice as a secondary option on Windows

        # ------------------ LibreOffice path (non-Windows or fallback) ------------------
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if not soffice:
            _log(
                "error",
                "LibreOffice 'soffice' not found. Install LibreOffice (in Docker image or host) "
                "or enable COM conversion on Windows (EMTAC_ENABLE_PPTX_TO_PDF=1)."
            )
            return None

        in_abs = os.path.abspath(pptx_file)
        out_dir_abs = os.path.abspath(out_dir)

        cmd = [
            soffice, "--headless",
            "--convert-to", "pdf",
            "--outdir", out_dir_abs,
            in_abs
        ]
        _log("info", f"Converting via LibreOffice: {' '.join(cmd)}")

        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec
            )
            _log("debug", f"LibreOffice stdout:\n{proc.stdout}")
            if proc.returncode != 0:
                _log("error", f"LibreOffice failed (code={proc.returncode}). stderr:\n{proc.stderr}")
                return None
        except subprocess.TimeoutExpired:
            _log("error", f"LibreOffice timed out after {timeout_sec}s")
            return None
        except Exception as e:
            _log("error", f"LibreOffice invocation error: {e}")
            return None

        # LibreOffice writes <stem>.pdf into out_dir_abs
        stem = os.path.splitext(os.path.basename(in_abs))[0]
        produced = os.path.join(out_dir_abs, stem + ".pdf")

        # In rare cases the name can differ slightly; fall back to “newest .pdf in out_dir”
        if not os.path.exists(produced):
            pdfs = [
                os.path.join(out_dir_abs, f) for f in os.listdir(out_dir_abs)
                if f.lower().endswith(".pdf")
            ]
            produced = max(pdfs, key=lambda p: os.path.getmtime(p)) if pdfs else None

        if not produced or not os.path.exists(produced):
            _log("error", "Converted PDF not found after LibreOffice conversion")
            return None

        # Move/rename to the exact requested `pdf_file` if needed
        if os.path.abspath(produced) != os.path.abspath(pdf_file):
            try:
                # Replace atomically if target exists
                os.replace(produced, pdf_file)
            except Exception as mv_err:
                _log("warning", f"Could not rename produced PDF; using {produced}. Reason: {mv_err}")
                pdf_file = produced

        _log("info", f"PPTX converted to PDF via LibreOffice: {pdf_file}")
        return pdf_file

    except Exception as e:
        _log("error", f"Unexpected PPTX->PDF failure: {e}")
        return None

upload_powerpoint_bp = Blueprint('upload_powerpoint_bp', __name__)

@upload_powerpoint_bp.route('/upload_powerpoint', methods=['POST'])
def upload_powerpoint():
    try:
        title = request.form.get('title')
        area = request.form.get('area')
        equipment_group = request.form.get('equipment_group')
        model = request.form.get('model')
        asset_number = request.form.get('asset_number')
        location = request.form.get('location')
        site_location = request.form.get('site_location')  # Get site_location from the request
        description = request.form.get('description')
        ppt_file = request.files.get('powerpoint')

        # Print the form data to the console for debugging
        print(f"title: {title}")
        print(f"area: {area}")
        print(f"equipment_group: {equipment_group}")
        print(f"model: {model}")
        print(f"asset_number: {asset_number}")
        print(f"location: {location}")
        print(f"site_location: {site_location}")
        print(f"description: {description}")

        # Log the form data
        logger.debug(f"title: {title}")
        logger.debug(f"area: {area}")
        logger.debug(f"equipment_group: {equipment_group}")
        logger.debug(f"model: {model}")
        logger.debug(f"asset_number: {asset_number}")
        logger.debug(f"location: {location}")
        logger.debug(f"site_location: {site_location}")
        logger.debug(f"description: {description}")

        if not title:
            filename = secure_filename(ppt_file.filename)
            title = os.path.splitext(filename)[0]

        if ppt_file is None:
            return "No PowerPoint file provided", 400

        if not os.path.exists(PPT2PDF_PPT_FILES_PROCESS):
            os.makedirs(PPT2PDF_PPT_FILES_PROCESS)

        ppt_filename = secure_filename(ppt_file.filename)
        ppt_path = os.path.join(PPT2PDF_PPT_FILES_PROCESS, ppt_filename)
        ppt_file.save(ppt_path)

        pdf_filename = ppt_filename.replace(".pptx", ".pdf")
        pdf_file_path = os.path.join(PPT2PDF_PDF_FILES_PROCESS, pdf_filename)
        convert_pptx_to_pdf(ppt_path, pdf_file_path)

        # Convert form values to integers or None
        area_id = int(area) if area else None
        equipment_group_id = int(equipment_group) if equipment_group else None
        model_id = int(model) if model else None
        asset_number_id = int(asset_number) if asset_number else None
        location_id = int(location) if location else None

        # Print the values to the console and log them
        print(f"area_id: {area_id}, equipment_group_id: {equipment_group_id}, model_id: {model_id}, asset_number_id: {asset_number_id}, location_id: {location_id}, site_location: {site_location}")
        logger.debug(f"area_id: {area_id}, equipment_group_id: {equipment_group_id}, model_id: {model_id}, asset_number_id: {asset_number_id}, location_id: {location_id}, site_location: {site_location}")

        logger.debug(f"Creating position with parameters - area: {area}, equipment_group: {equipment_group}, model: {model}, asset_number: {asset_number}, location: {location}, site_location: {site_location}")
        position_id = create_position(area, equipment_group, model, asset_number, location, site_location, )
        logger.debug(f"Position ID created: {position_id}")

        if not position_id:
            logger.error("Failed to create position")
            return "Failed to create position", 500

        complete_document_id, success = add_document_to_db(title, pdf_file_path, position_id)
        logger.info(f"Complete document ID: {complete_document_id}, Success: {success}")

        if success:
            with Session() as session:
                new_powerpoint_id = add_powerpoint_to_db(session, title, ppt_path, pdf_file_path, complete_document_id, description)
                if not new_powerpoint_id:
                    return "Failed to add PowerPoint to the database", 500
        else:
            return "Failed to add document to the database", 500

        if success:
            return "Document uploaded and processed successfully"
        else:
            return "Failed to process the document", 500

    except Exception as e:
        logger.error(f"Error during PowerPoint upload and conversion: {str(e)}")
        return f"Error during PowerPoint upload and conversion: {str(e)}", 500
