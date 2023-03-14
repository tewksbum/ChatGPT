import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os

##################################################
##################################################
##################################################
##################################################

 import requests
from bs4 import BeautifulSoup

def get_links(url):
    # Make a request to the landing page
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all the hyperlinks in the landing page
    links = soup.find_all("a")

    # Extract the href attribute of each hyperlink
    urls = [link.get("href") for link in links]

    # Filter out None and empty strings
    urls = list(filter(None, urls))

    # Make the hyperlinks absolute URLs
    urls = [requests.compat.urljoin(url, u) for u in urls]

    return urls

def get_all_links(url):
    # Get the links from the landing page
    links = get_links(url)

    # Recursively get the links from all the pages linked to from the landing page
    for link in links:
        try:
            # Get the links from the linked page
            linked_links = get_all_links(link)

            # Add the linked links to the list of links
            links.extend(linked_links)
        except:
            pass

    return links

##################################################
##################################################
##################################################
##################################################

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

# Define root domain to crawl
domain = "patspulpit.com"
full_url = "https://www.patspulpit.com/"

# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        
        #c-entry-box--compact__body
        #data-chorus-optimize-field
        #c-entry-box--compact

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

# Function to get the hyperlinks from a URL
def get_hyperlinks(url):
    
    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []
            
            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            #if re.search("/news/", clean_link) or re.search("/team/stats/", clean_link):
                clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))


def crawl(url):
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files
    if not os.path.exists("text/"):
            os.mkdir("text/")

    if not os.path.exists("text/"+local_domain+"/"):
            os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
            os.mkdir("processed")

    # While the queue is not empty, continue crawling
    while queue:

        # Get the next URL from the queue
        url = queue.pop()
        print(url) # for debugging and to see the progress

        # Save text from the url to a <url>.txt file
        with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w") as f:

            # Get the text from the URL using BeautifulSoup
            soup = BeautifulSoup(requests.get(url).text, "html.parser")
            
            # Get the text but remove the tags
            text = soup.get_text()
            
            # If the crawler gets to a page that requires JavaScript, it will stop the crawl
            if ("You need to enable JavaScript to run this app." in text):
                print("Unable to parse page " + url + " due to JavaScript being required")
            
            title = ""
            description = ""
            body = ""
            
            
            if soup.find(class_="c-page-title"):
                title = soup.find(class_="nfl-c-page").get_text()
                if soup.find(class_="c-entry-summary"):
                    description = soup.find(class_="c-entry-summary").get_text()
                if soup.find(class_="c-entry-content"):
                    body = soup.find(class_="c-entry-content").get_text()
                # Otherwise, write the text to the file in the text directory
                bodytext = "TITLE:" + str(title) + ";DESC=" + str(description) + ";BODY=" + str(body) + " END"
                print(bodytext)
                f.write(bodytext)

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)

print('hello')
crawl(full_url)
print('world')