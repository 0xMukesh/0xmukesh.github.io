import json
from datetime import datetime, timezone
from email.utils import format_datetime

site = "https://mukesh.0xc84.fyi"
feed_title = "Mukesh"
feed_description = "My random recreational programming musings"
items = []

with open("./public/data/blogs.json", "r") as f:
    blogs = json.load(f)

    for blog in blogs:
        if blog["draft"]:
            continue

        blog_title = blog["title"]
        blog_description = blog[
            "feed_description"
        ]  # to avoid clashing with `description` which is used for `projects.json`
        url = f"{site}{blog['url']}"  # `url` field already has a leading slash
        published_at = format_datetime(  # RFC822 compliant date-times. ref: https://stackoverflow.com/a/22905935
            datetime.strptime(blog["date"], "%B %d, %Y").replace(tzinfo=timezone.utc),
            usegmt=True,
        )

        items.append(
            f"""
<item>
  <title>{blog_title}</title>
  <link>{url}</link>
  <description>{blog_description}</description>
  <pubDate>{published_at}</pubDate>
</item>"""
        )

with open("feed.xml", "w") as f:
    f.write(
        f"""<rss version="2.0">
  <channel>
    <title>{feed_title}</title>
    <description>{feed_description}</description>
    <link>{site}/blog</link>
    {"".join(items)}
  </channel>
</rss>"""
    )
