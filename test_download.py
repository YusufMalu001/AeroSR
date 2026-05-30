import requests
import re
import os
import sys

def download_gdrive_file(file_id, dest_path):
    print(f"Starting download for file ID: {file_id} -> {dest_path}")
    session = requests.Session()
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    r = session.get(url)
    
    if r.status_code != 200:
        print(f"Failed to fetch initial URL: {r.status_code}")
        return False
        
    html = r.text
    # Search for action URL in form
    action_match = re.search(r'action="([^"]*)"', html)
    if action_match:
        action = action_match.group(1)
        # Parse all hidden input fields
        params = {}
        for name, value in re.findall(r'<input type="hidden" name="([^"]*)" value="([^"]*)"', html):
            params[name] = value
        
        # In case the regex didn't catch everything, ensure confirm=t is there
        if 'confirm' not in params:
            params['confirm'] = 't'
            
        print(f"Warning page detected. Downloading using action: {action} with params: {params}")
        r = session.get(action, params=params, stream=True)
    else:
        # No warning page, download directly
        print("Direct download detected (no warning page).")
        r = session.get(url, stream=True)
        
    if r.status_code != 200:
        print(f"Download request failed with status: {r.status_code}")
        return False
        
    # Write to destination
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                
    print(f"Downloaded successfully. Size: {os.path.getsize(dest_path) / 1e6:.2f} MB")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 2:
        download_gdrive_file(sys.argv[1], sys.argv[2])
    else:
        # Default test with Annotations.zip
        download_gdrive_file("1KoQzqR20qvIXDf1qsXCHGxD003IPmXMw", "annotations_test.zip")
