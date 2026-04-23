from pathlib import Path

import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"

    # async def start(self):
    #     urls = [
    #         "https://quotes.toscrape.com/page/1/",
    #         "https://quotes.toscrape.com/page/2/",
    #     ]

    #     for url in urls:
    #         yield scrapy.Request(url=url, callback=self.parse)

    start_urls = [
        "https://quotes.toscrape.com/page/1/",
        "https://quotes.toscrape.com/page/2/",]
    
    def parse(self, response : scrapy.Request):
        # page = response.url.split("/")[-2]
        # filename = f"quotes-{page}.html"
        # Path(filename).write_bytes(response.body)
        # self.log(f"Saved file: {filename}")
        for quote in response.css("div.quote"):
            yield {
                "text" : quote.css("span.text::text").get(),
                "author" : quote.css("small.author::text").get(),
                "tags" : quote.css("div.tags a.tag::text").getall(),
            }
