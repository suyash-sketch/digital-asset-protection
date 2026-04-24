import scrapy
import urllib.parse
import urllib.request
import re # ADD THIS IMPORT!
from pathlib import Path
from scrapy_playwright.page import PageMethod

class InstagramSpider(scrapy.Spider):
    name = "instagram"

    custom_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    playwright_cookies = [
        {'name': 'sessionid',  'value':"45793584147%3AvhiPhQBKDKUtAH%3A25%3AAYg0fVkt1Bjd0P4lIc8-OH2KoEZl9pP42OY1dhdD8g", 'domain': '.instagram.com', 'path': '/'},
        {'name': 'csrftoken', 'value': 'sQ81cz2NYPzpzmJsM3YV5ck65dYOKt0p', 'domain': '.instagram.com', 'path': '/'},
    ]

    async def start(self):
        url = "https://www.instagram.com/suuu.yash/"
        yield scrapy.Request(
            url,
            headers=self.custom_headers,
            meta={
                "playwright": True,
                "playwright_context_kwargs": {
                    "storage_state": {"cookies": self.playwright_cookies}
                },
                "playwright_page_methods": [
                    PageMethod("wait_for_timeout", 5000), 
                    PageMethod("evaluate", "window.scrollBy(0, document.body.scrollHeight)"),
                    PageMethod("wait_for_timeout", 3000), 
                ],
            },
        )

    def parse(self, response):
        self.logger.info(f"Successfully reached Grid: {response.url}")
        
        post_links = response.css('main a[href*="/p/"]::attr(href), main a[href*="/reel/"]::attr(href)').getall()
        post_links = list(set(post_links)) 
        
        self.logger.info(f"Found {len(post_links)} posts. Visiting each one to check for videos...")
        
        for link in post_links:
            post_url = response.urljoin(link)
            yield scrapy.Request(
                url=post_url, 
                callback=self.parse_video_page,
                headers=self.custom_headers,
                meta={
                    "playwright": True,
                    "playwright_context_kwargs": {
                        "storage_state": {"cookies": self.playwright_cookies}
                    },
                    "playwright_page_methods": [
                        PageMethod("wait_for_timeout", 4000), 
                    ],
                }
            )

    def parse_video_page(self, response):
        self.logger.info(f"Checking for media on: {response.url}")
        media_url = None
        is_video = False
        
        # --- 1. Check for Video ---
        meta_vid = response.css('meta[property="og:video"]::attr(content)').get()
        if meta_vid and not meta_vid.startswith('blob:'):
            media_url = meta_vid
            is_video = True
            
        if not media_url:
            match = re.search(r'"video_url":"([^"]+)"', response.text)
            if match:
                media_url = match.group(1).replace('\\/', '/')
                is_video = True

        # --- 2. Fallback to High-Res Image (If no video, or if it's blob-protected) ---
        if not media_url:
            self.logger.info("No direct video found (Photo post or Blob-protected). Grabbing high-res photo...")
            
            # Try to get the highest quality OpenGraph image meta tag
            meta_img = response.css('meta[property="og:image"]::attr(content)').get()
            if meta_img:
                media_url = meta_img
        
        # Absolute last resort fallback for images
        if not media_url:
            img_tag = response.css('article img::attr(src)').get()
            if img_tag and not img_tag.startswith('data:image'):
                media_url = img_tag

        if not media_url:
            self.logger.warning(f"Could not extract any downloadable media from {response.url}")
            return
            
        # --- 3. Download the Media ---
        url_path = urllib.parse.urlparse(media_url).path
        filename = url_path.split("/")[-1]
        
        # Assign proper extensions and folders based on what we found
        if is_video and not filename.endswith('.mp4'):
            filename += '.mp4'
        elif not is_video and not filename.endswith(('.jpg', '.png', '.webp')):
            filename += '.jpg'
            
        folder_name = "videos" if is_video else "images"
        path = Path(folder_name) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            req = urllib.request.Request(media_url, headers=self.custom_headers)
            with urllib.request.urlopen(req) as media_response, open(path, "wb") as f:
                f.write(media_response.read())
            
            # This logs either "Saved VIDEO..." or "Saved IMAGE..."
            self.logger.info(f"Saved {folder_name[:-1].upper()} successfully: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to download {filename}: {e}")