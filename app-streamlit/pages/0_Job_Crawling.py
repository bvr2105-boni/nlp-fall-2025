import streamlit as st
import pandas as pd
import csv
import os
from datetime import datetime
import random
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TimeFilters, TypeFilters, ExperienceLevelFilters, \
    OnSiteOrRemoteFilters, SalaryBaseFilters
import logging
from components.header import setup_page

# Suppress logging
logging.getLogger().setLevel(logging.ERROR)

st.title("Job Data Collection")

st.markdown("""
<div style="margin-bottom: 2rem;">
    <p style="color: #6b7280; font-size: 1.1rem; line-height: 1.6;">
        Collect job postings from LinkedIn to build your analysis dataset.
        Choose job titles and let our automated scraper gather comprehensive job information for market insights.
    </p>
</div>
""", unsafe_allow_html=True)

# Define job titles
job_titles = [
    'Digital Marketing Specialist', 'Business Development Manager',
    'Quality Assurance Analyst', 'Systems Administrator', 'Database Administrator', 'Cybersecurity Analyst', 'DevOps Engineer',
    'Mobile App Developer', 'Cloud Solutions Architect', 'Technical Support Engineer', 'SEO Specialist', 'Social Media Manager',
    'Content Marketing Manager', 'E-commerce Manager', 'Brand Manager', 'Public Relations Specialist', 'Event Coordinator',
    'Logistics Manager', 'Supply Chain Analyst', 'Operations Analyst', 'Risk Manager', 'Compliance Officer', 'Auditor', 'Tax Specialist',
    'Investment Analyst', 'Portfolio Manager', 'Real Estate Agent', 'Insurance Underwriter', 'Claims Adjuster', 'Actuary', 'Loan Officer', 'Credit Analyst', 'Treasury Analyst', 'Financial Planner',
    'Marketing Analyst', 'Market Research Analyst', 'Advertising Manager', 'Media Planner', 'Copywriter', 'Video Producer', 'Animator', 'Illustrator', 'Interior Designer', 'Architect',
    'Civil Engineer', 'Mechanical Engineer', 'Electrical Engineer', 'Chemical Engineer', 'Environmental Engineer', 'Biomedical Engineer', 'Industrial Engineer', 'Aerospace Engineer', 'Petroleum Engineer', 'Nuclear Engineer',
    'Pharmacist', 'Nurse Practitioner', 'Physician Assistant', 'Medical Laboratory Technician', 'Radiologic Technologist', 'Physical Therapist', 'Occupational Therapist', 'Speech-Language Pathologist', 'Dietitian', 'Respiratory Therapist',
    'Teacher', 'School Counselor', 'Librarian', 'Social Worker', 'Psychologist', 'Counselor', 'Therapist', 'Coach', 'Trainer', 'Recruiter'
]

# User input
selected_titles = st.multiselect(
    "Select job titles to scrape:",
    job_titles,
    default=['Data Scientist', 'Software Engineer'] if 'Data Scientist' in job_titles and 'Software Engineer' in job_titles else job_titles[:2],
    help="Select one or more job titles. Note: Scraping many titles may take time and could be rate-limited by LinkedIn."
)

max_jobs = st.slider(
    "Maximum jobs per title:",
    min_value=1,
    max_value=100,
    value=2,
    step=1,
    help="Limit the number of jobs to scrape per selected title."
)

# LinkedIn authentication
st.subheader("🔐 LinkedIn Authentication (Optional)")
st.markdown("""
**To get your `li_at` cookie:**

1. Log into LinkedIn in Chrome
2. Open Developer Tools (F12)
3. Go to Application → Cookies → `https://www.linkedin.com`
4. Find and copy the `li_at` cookie value
""")

li_at_cookie = st.text_input(
    "li_at Cookie",
    type="password",
    help="Enter your LinkedIn li_at cookie for authenticated scraping. This helps avoid rate limiting."
)

if li_at_cookie:
    st.info(f"Cookie set: {li_at_cookie[:10]}... (length: {len(li_at_cookie)})")
else:
    st.info("No cookie set. Scraping may be rate-limited.")

if st.button("Start Scraping", type="primary"):
    if not selected_titles:
        st.error("Please select at least one job title.")
    else:
        st.info(f"Starting to scrape jobs for: {', '.join(selected_titles)}")
        
        # Initialize data collection
        all_jobs = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_titles = len(selected_titles)
        
        for idx, title in enumerate(selected_titles):
            status_text.text(f"Scraping jobs for: {title} ({idx+1}/{total_titles})")
            
            # Collect data for this title
            jobs_data = []
            
            def on_data(data: EventData):
                jobs_data.append({
                    'Job Title': data.title,
                    'Company': data.company,
                    'Company Link': data.company_link,
                    'Date': data.date,
                    'Date Text': data.date_text,
                    'Job Link': data.link,
                    'Insights': ', '.join(data.insights) if data.insights else '',
                    'Description Length': len(data.description),
                    'Description': data.description.replace('\n', ' ').replace('\r', ' ') if data.description else ''
                })
            
            def on_metrics(metrics: EventMetrics):
                pass
            
            def on_error(error):
                st.warning(f"Error scraping {title}: {error}")
            
            def on_end():
                pass
            
            # Initialize scraper
            scraper = LinkedinScraper(
                chrome_executable_path=None,
                chrome_binary_location=None,
                chrome_options=None,
                headless=True,
                max_workers=1,
                slow_mo=2.0,
                page_load_timeout=40
            )
            
            # Add event listeners
            scraper.on(Events.DATA, on_data)
            scraper.on(Events.ERROR, on_error)
            scraper.on(Events.END, on_end)
            
            # Create query for this title
            queries = [
                Query(
                    query=title,
                    options=QueryOptions(
                        locations=['United States'],
                        apply_link=True,
                        skip_promoted_jobs=True,
                        page_offset=0,
                        limit=max_jobs,
                        filters=QueryFilters(
                            relevance=RelevanceFilters.RECENT,
                            time=TimeFilters.WEEK,
                            type=[TypeFilters.FULL_TIME, TypeFilters.INTERNSHIP],
                            on_site_or_remote=[OnSiteOrRemoteFilters.REMOTE],
                            experience=[ExperienceLevelFilters.MID_SENIOR],
                            base_salary=SalaryBaseFilters.SALARY_100K
                        )
                    )
                )
            ]
            
            # Run scraper
            try:
                scraper.run(queries)
                all_jobs.extend(jobs_data)
            except Exception as e:
                st.error(f"Failed to scrape {title}: {str(e)}")
            
            # Update progress
            progress_bar.progress((idx + 1) / total_titles)
            
            # Small delay between titles to avoid rate limiting
            if idx < total_titles - 1:
                import time
                time.sleep(random.uniform(5, 15))
        
        status_text.text("Scraping completed!")
        progress_bar.empty()
        
        if all_jobs:
            # Convert to DataFrame
            df = pd.DataFrame(all_jobs)
            
            st.success(f"Successfully scraped {len(df)} jobs!")
            
            # Display summary
            st.subheader("📊 Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Jobs", len(df))
            with col2:
                st.metric("Unique Companies", df['Company'].nunique())
            with col3:
                st.metric("Average Description Length", f"{df['Description Length'].mean():.0f} chars")
            
            # Display data
            st.subheader("📋 Scraped Jobs")
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"linkedin_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv"
            )
            
            # Save to workspace
            workspace_path = st.session_state.get('workspace_path')
            if workspace_path:
                scraps_path = os.path.join(workspace_path, "scraps")
                if not os.path.exists(scraps_path):
                    os.makedirs(scraps_path)
                csv_filename = f"scraps/linkedin_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(csv_filename, index=False)
                st.info(f"Data also saved to: {csv_filename}")
        else:
            st.warning("No jobs were scraped. This might be due to LinkedIn rate limiting or network issues.")

st.markdown("---")
st.markdown("""
**Note:** 
- LinkedIn may rate-limit scraping activities. If you encounter errors, try again later or reduce the number of jobs.
- The scraper uses headless Chrome, so Chrome must be installed on the system.
- Scraping may take several minutes depending on the number of jobs and titles selected.
""")