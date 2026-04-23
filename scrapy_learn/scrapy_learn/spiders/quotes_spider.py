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
        self.logger.info(f"Checking for video on: {response.url}")
        video_src = None
        
        # 1. Try Meta Tag (Fastest, if it exists)
        meta_vid = response.css('meta[property="og:video"]::attr(content)').get()
        if meta_vid and not meta_vid.startswith('blob:'):
            video_src = meta_vid
            
        # 2. Try Regex on the raw JSON (The most reliable for Reels!)
        if not video_src:
            match = re.search(r'"video_url":"([^"]+)"', response.text)
            if match:
                video_src = match.group(1).replace('\\/', '/')
                self.logger.info("Found video URL via hidden JSON data!")
                
        # 3. Try standard Video tag (Fallback)
        if not video_src:
            vid_tag = response.css('video::attr(src)').get()
            if vid_tag and not vid_tag.startswith('blob:'):
                video_src = vid_tag
                
        # 4. Try Video Source tag (Fallback)
        if not video_src:
            src_tag = response.css('video source::attr(src)').get()
            if src_tag and not src_tag.startswith('blob:'):
                video_src = src_tag

        # Final check
        if not video_src:
            self.logger.info(f"No usable video found on {response.url}. It is either a photo or fully blob-protected.")
            return
            
        # --- Download the .mp4 file ---
        url_path = urllib.parse.urlparse(video_src).path
        filename = url_path.split("/")[-1]
        
        if not filename.endswith('.mp4'):
            filename += '.mp4'
            
        path = Path("videos") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            req = urllib.request.Request(video_src, headers=self.custom_headers)
            with urllib.request.urlopen(req) as vid_response, open(path, "wb") as f:
                f.write(vid_response.read())
            
            self.logger.info(f"Saved VIDEO successfully: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to download video {filename}: {e}")