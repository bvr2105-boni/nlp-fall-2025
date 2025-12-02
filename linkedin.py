from asyncio import sleep
import logging
import csv
import os
from datetime import datetime
import random
import argparse
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TimeFilters, TypeFilters, ExperienceLevelFilters, \
    OnSiteOrRemoteFilters, SalaryBaseFilters

# Change root logger level (default is WARN)
logging.basicConfig(level=logging.INFO)


# CLI arguments
parser = argparse.ArgumentParser(description="Scrape LinkedIn jobs and write results to CSV.")
parser.add_argument("--title", action="append", dest="titles", help="Job title to search. Repeat for multiple titles.")
parser.add_argument("--limit", type=int, default=1, help="Max number of jobs to scrape for each title.")
args = parser.parse_args()

# Define job titles to scrape
job_titles = args.titles if args.titles else ['Risk Manager']

for title in job_titles:

        # CSV file setup
    csv_filename = f"linkedin_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)

    # Write CSV header
    csv_writer.writerow([
        'Job Title', 
        'Company', 
        'Company Link', 
        'Date', 
        'Date Text', 
        'Job Link', 
        'Insights', 
        'Description Length',
        'Description'
    ])

    # Fired once for each successfully processed job
    def on_data(data: EventData):
        print('[ON_DATA]', data.title, data.company, data.company_link, data.date, data.date_text, data.link, data.insights,
            len(data.description))
        
        # Write data to CSV
        csv_writer.writerow([
            data.title,
            data.company,
            data.company_link,
            data.date,
            data.date_text,
            data.link,
            ', '.join(data.insights) if data.insights else '',
            len(data.description),
            data.description.replace('\n', ' ').replace('\r', ' ') if data.description else ''
        ])


    # Fired once for each page (25 jobs)
    def on_metrics(metrics: EventMetrics):
        print('[ON_METRICS]', str(metrics))


    def on_error(error):
        print('[ON_ERROR]', error)


    def on_end():
        print('[ON_END]')
        csv_file.close()
        print(f'Data saved to {csv_filename}')


    scraper = LinkedinScraper(
        chrome_executable_path=None,  # Custom Chrome executable path (e.g. /foo/bar/bin/chromedriver)
        chrome_binary_location=None,  # Custom path to Chrome/Chromium binary (e.g. /foo/bar/chrome-mac/Chromium.app/Contents/MacOS/Chromium)
        chrome_options=None,  # Custom Chrome options here
        headless=True,  # Overrides headless mode only if chrome_options is None
        max_workers=1,  # How many threads will be spawned to run queries concurrently (one Chrome driver for each thread)
        slow_mo=2.0,  # Slow down the scraper to avoid 'Too many requests 429' errors (5 seconds delay between requests)
        page_load_timeout=40  # Page load timeout (in seconds)    
    )

    # Add event listeners
    scraper.on(Events.DATA, on_data)
    scraper.on(Events.ERROR, on_error)
    scraper.on(Events.END, on_end)

    queries = [
        Query(
            options=QueryOptions(
                limit=1  # Limit the number of jobs to scrape.            
            )
        ),
        Query(
            query=title,
            options=QueryOptions(
                locations=['United States'],
                apply_link=True,  # Try to extract apply link (easy applies are skipped). If set to True, scraping is slower because an additional page must be navigated. Default to False.
                skip_promoted_jobs=True,  # Skip promoted jobs. Default to False.
                page_offset=0,  # How many pages to skip
                limit=args.limit,
                filters=QueryFilters(
                    company_jobs_url='https://www.linkedin.com/jobs/search/?currentJobId=4297199509&f_C=11448%2C1035%2C1418841%2C10073178%2C11206713%2C1148098%2C1386954%2C165397%2C18086638%2C1889423%2C19053704%2C19537%2C2270931%2C2446424%2C263515%2C30203%2C3178875%2C3238203%2C3290211%2C3641570%2C3763403%2C5097047%2C5607466%2C589037%2C692068&geoId=92000000&origin=JOB_SEARCH_PAGE_JOB_FILTER',  # Filter by companies.                
                    relevance=RelevanceFilters.RECENT,
                    time=TimeFilters.WEEK,
                    type=[TypeFilters.FULL_TIME, TypeFilters.INTERNSHIP],
                    on_site_or_remote=[OnSiteOrRemoteFilters.REMOTE],
                    experience=[ExperienceLevelFilters.MID_SENIOR],
                    base_salary=SalaryBaseFilters.SALARY_100K
                )
            )
        ),
    ]

    scraper.run(queries)

    # sleep(random.randint(60, 240))  # Wait for a few seconds before starting the next title scrape to avoid rate limiting
