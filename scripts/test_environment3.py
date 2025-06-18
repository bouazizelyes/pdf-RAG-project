import fitz  # PyMuPDF is imported as 'fitz'
import os

print("--- Testing PyMuPDF (fitz) Library ---")

try:
    # 1. Check the version correctly
    # The version is stored in a tuple, e.g., ('1.24.4', '1.24.0', ...)
    print(f"✅ PyMuPDF version: {fitz.version[0]}")

    # 2. Perform a real operation: create a PDF, save it, and read it back
    file_name = "test_document.pdf"
    test_text = "Hello, RAG world! This PDF was created by PyMuPDF."

    # Create a new, blank PDF document in memory
    doc = fitz.open()
    # Add a page
    page = doc.new_page()
    # Insert text into the page
    page.insert_text((50, 72), test_text)
    
    # Save the document to a file
    doc.save(file_name)
    doc.close()
    print(f"✅ Successfully created and saved '{file_name}'")

    # 3. Open the saved PDF and verify its content
    doc_read = fitz.open(file_name)
    page_read = doc_read[0]
    extracted_text = page_read.get_text()
    doc_read.close()

    if test_text in extracted_text:
        print(f"✅ Successfully read back and verified the text.")
    else:
        print(f"❌ Text verification failed!")
        print(f"   - Expected: '{test_text}'")
        print(f"   - Got: '{extracted_text}'")

    # Clean up the test file
    os.remove(file_name)
    print("✅ Cleaned up the test file.")

    print("\n--- PyMuPDF Test Passed! ---")

except Exception as e:
    print(f"\n--- ❌ PyMuPDF Test FAILED ---")
    print(f"An error occurred: {e}")
