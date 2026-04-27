import scrapy
import urllib.parse
import urllib.request
import re
from pathlib import Path
from scrapy_playwright.page import PageMethod

class InstagramSpider(scrapy.Spider):
    name = "instagram"

    custom_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    
    playwright_cookies = [
        {'name': 'sessionid',  'value':"46746482287%3AMRj3PkK2SeSXwK%3A20%3AAYiFSvD6pGX_lm3Idb-DSMZasUpMh3tAfg2UIdOVog", 'domain': '.instagram.com', 'path': '/'},
        {'name': 'csrftoken', 'value': 'cicu1D8-zrngKLjhkTw6g6', 'domain': '.instagram.com', 'path': '/'},
    ]

    @staticmethod
    def _decode_ig_url(raw_url):
        if not raw_url:
            return None
        return raw_url.replace("\\/", "/").replace("\\u0026", "&")

    def _extract_best_video_url(self, response):
        text = response.text
        mp4_matches = re.findall(r'https:\\/\\/[^"]+?\.mp4[^"]*', text)
        if not mp4_matches:
            mp4_matches = re.findall(r'"video_url":"([^"]+)"', text)
        if not mp4_matches:
            return None
        return self._decode_ig_url(mp4_matches[0])

    def _extract_best_image_url(self, response):
        text = response.text

        srcset_candidates = []

        for srcset in response.css("article img::attr(srcset)").getall():
            for entry in srcset.split(","):
                part = entry.strip()
                if not part:
                    continue
                fields = part.split()
                if not fields:
                    continue
                candidate_url = fields[0]
                width = 0
                if len(fields) > 1 and fields[1].endswith("w"):
                    try:
                        width = int(fields[1][:-1])
                    except ValueError:
                        width = 0
                srcset_candidates.append((width, candidate_url))

        if srcset_candidates:
            # Pick the largest image from the post's own media srcset.
            best_srcset = max(srcset_candidates, key=lambda x: x[0])
            return self._decode_ig_url(best_srcset[1])

        versioned_candidates = re.findall(
            r'"url":"([^"]+)","width":(\d+),"height":(\d+)',
            text,
        )
        if versioned_candidates:
            # Prefer large post-media candidates over small avatar/preview images.
            large_candidates = [
                c for c in versioned_candidates if int(c[1]) >= 700 or int(c[2]) >= 700
            ]
            pool = large_candidates if large_candidates else versioned_candidates
            best_versioned = max(pool, key=lambda c: int(c[1]) * int(c[2]))
            return self._decode_ig_url(best_versioned[0])

        # Instagram often exposes multiple image sizes in display_resources;
        # pick the largest one to avoid cropped/low-res previews.
        resources = re.findall(
            r'"src":"([^"]+)","config_width":(\d+),"config_height":(\d+)',
            text,
        )
        if resources:
            best = max(resources, key=lambda r: int(r[1]) * int(r[2]))
            return self._decode_ig_url(best[0])

        display_url = re.search(r'"display_url":"([^"]+)"', text)
        if display_url:
            return self._decode_ig_url(display_url.group(1))

        meta_img = response.css('meta[property="og:image"]::attr(content)').get()
        if meta_img:
            return meta_img

        img_tag = response.css("article img::attr(src)").get()
        if img_tag and not img_tag.startswith("data:image"):
            return img_tag

        return None

    async def start(self):
        url = "https://www.instagram.com/kharadesarthak/"
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
        media_url = self._extract_best_video_url(response)
        is_video = bool(media_url)

        if not media_url:
            self.logger.info("No downloadable video found; trying full-resolution image URL...")
            media_url = self._extract_best_image_url(response)

        if not media_url:
            self.logger.warning(f"Could not extract any downloadable media from {response.url}")
            return
        self.logger.info(f"Selected media URL: {media_url}")
            
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