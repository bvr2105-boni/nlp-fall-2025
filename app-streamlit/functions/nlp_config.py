"""Shared configuration and constants for NLP analytics (skills, stopwords, etc.)."""

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Extended multilingual stopwords for topic modeling (LDA / LSA)
# Includes English plus common FR/ES/DE/NL/PT/IT function words and job-posting boilerplate.
EXTRA_MULTILINGUAL_STOPWORDS = {
    # French
    "en", "et", "la", "des", "les", "que", "du", "dans", "avec", "pour", "nous", "vous", "une", "au",
    # Spanish
    "el", "para", "una", "del", "las", "los", "de", "y", "con", "por", "experiencia", "como", "nos",
    # German / Dutch / mixed
    "und", "mit", "der", "die", "das", "den", "im", "zu", "fur", "für", "von", "auf",
    "wir", "wij", "zijn", "wat",
    "je", "jij", "jouw", "jij", "jou", "onze", "ons",
    "een", "van", "het", "voor", "met", "op", "te", "om", "bij", "als", "aan",
    # Portuguese / Italian
    "em", "com", "da", "de", "na", "nos", "um", "uma", "ou", "mais",
    "voce", "você", "ao", "os", "dos", "di", "il", "le", "se", "al",
    # Generic people / team words that are mostly boilerplate
    "pessoas", "equipe", "equipo",
}


# Final stopword list used by topic modeling vectorizers
TOPIC_MODEL_STOPWORDS = sorted(ENGLISH_STOP_WORDS.union(EXTRA_MULTILINGUAL_STOPWORDS))


# --- Skill configuration ---

# Base master skill list from NER notebook (530+ skills)
BASE_MASTER_SKILL_LIST = [
    # Technical / Programming
    "python", "r", "java", "javascript", "typescript",
    "c++", "c#", "scala", "go", "matlab",
    "bash", "shell scripting",
    "software engineering", "software development",
    "full stack development", "frontend development", "backend development",
    "api design", "rest apis", "microservices",
    "distributed systems", "scalable systems",
    "cloud infrastructure", "cloud computing", "cloud native", "cloud platforms",

    # Data Analytics
    "sql", "nosql", "postgresql", "mysql", "oracle", "sqlite",
    "mongodb", "snowflake", "redshift", "bigquery", "azure sql",
    "data analysis", "data analytics", "statistical analysis",
    "business intelligence", "operational reporting",
    "process mapping", "requirements analysis",
    "risk management", "financial reporting",

    # Data Tools
    "pandas", "numpy", "scipy", "matplotlib", "seaborn",
    "plotly", "pyspark", "spark", "hadoop", "hive", "mapreduce", "jira",

    # Machine Learning
    "machine learning", "deep learning", "neural networks",
    "logistic regression", "linear regression", "random forest",
    "xgboost", "lightgbm", "catboost",
    "svm", "knn", "decision trees", "pca", "kmeans",
    "gradient boosting", "model tuning", "feature engineering",

    # NLP
    "nlp", "natural language processing", "topic modeling",
    "lda", "lsa", "keyword extraction",
    "named entity recognition", "text classification",
    "sentiment analysis", "embeddings", "bert", "word2vec",

    # Cloud
    "aws", "azure", "gcp", "docker", "kubernetes",
    "lambda", "ec2", "s3", "athena", "dynamodb",
    "databricks", "airflow", "cloud functions",

    # BI Tools
    "tableau", "power bi", "metabase", "looker", "qlik",
    "data visualization", "dashboard development",

    # ETL / Pipelines
    "etl", "elt", "data pipeline", "data ingestion",
    "data cleaning", "data transformation", "data integration",

    # Version Control & DevOps
    "git", "github", "gitlab", "bitbucket",
    "ci/cd", "jenkins",

    # Enterprise Tools
    "sap", "sap erp", "salesforce", "salesforce crm",
    "hubspot", "hubspot crm", "airtable", "jira", "confluence", "notion",

    # Business & Analytics Skills
    "business analysis", "requirements gathering",
    "market research", "competitive analysis",
    "financial analysis", "risk analysis", "cost analysis",
    "forecasting", "trend analysis", "variance analysis",
    "p&l management", "strategic planning",
    "business modeling", "stakeholder management",
    "reporting", "presentation development",
    "process improvement", "process optimization",
    "root cause analysis", "gap analysis",
    "workflow automation", "operational efficiency",
    "kpi analysis", "performance analysis",
    "customer segmentation", "persona development",
    "data-driven decision making",

    # Consulting skills
    "problem solving", "insights synthesis",
    "client communication", "proposal writing",
    "project scoping", "roadmap planning",
    "change management", "cross-functional collaboration",

    # Marketing / Sales
    "crm management", "lead generation", "pipeline management",
    "sales operations", "sales strategy", "sales forecasting",
    "revenue operations", "revops", "gtm strategy",
    "go-to-market", "account management",
    "client success", "customer retention", "digital marketing",
    "content marketing", "seo", "sem", "ppc", "email marketing",
    "campaign optimization", "social media analytics",

    # Marketing tools
    "marketing automation", "google analytics",
    "google ads", "mailchimp", "marketo",
    "outreach", "gong", "zoominfo",

    # RevOps Processes
    "validation rules", "crm integrations",
    "funnel analysis", "data stamping",

    # Product Skills
    "product management", "product analytics",
    "a/b testing", "experiment design",
    "feature prioritization", "user research", "ux research",
    "user stories", "agile", "scrum", "kanban",
    "roadmap development", "user journey mapping",
    "requirements documentation",
    "market sizing", "competitive positioning",

    # Finance & Operations Skills
    "fp&a", "financial modeling", "budgeting",
    "scenario analysis", "invoice processing",
    "billing operations", "revenue analysis",
    "cost optimization",

    # Operations & Supply Chain
    "supply chain management", "inventory management",
    "logistics", "procurement", "vendor management",
    "operations management", "kpi reporting",

    # Soft Skills
    "communication", "leadership", "teamwork",
    "collaboration", "critical thinking", "problem solving",
    "adaptability", "time management",
    "presentation skills", "negotiation",
    "public speaking", "project management",
    "detail oriented", "strategic thinking",
    "multitasking", "analytical thinking",
    "decision making", "organization skills",
    "attention to detail", "stakeholder communication",
    "conflict resolution", "problem-solving skills",
    "relationship building", "coaching", "mentoring",
]


# Extra skills / Tools (multi-industry) from NER notebook
EXTRA_SKILLS = [
    # Programming / tech
    "django", "flask", "fastapi",
    "react", "react native", "angular", "vue.js", "next.js",
    "node.js", "express.js",
    "php", "ruby", "ruby on rails",
    "swift", "kotlin", "objective-c",
    "c", "perl", "rust", "haskell",

    # Mobile / app
    "android development", "ios development",
    "xcode", "android studio",

    # Testing / QA
    "unit testing", "integration testing",
    "qa testing", "automation testing",
    "selenium", "cypress", "pytest", "junit",

    # Security / networking
    "network security", "firewall configuration",
    "penetration testing", "vulnerability assessment",
    "siem", "splunk", "wireshark",
    "ssl", "tls", "vpn",

    # Data analytics
    "excel", "microsoft excel",
    "vlookup", "pivot tables",
    "google sheets",
    "sql server", "db2",
    "sas", "stata", "spss",
    "power query", "power pivot",
    "mode analytics", "lookml",
    "amplitude", "mixpanel",
    "hex", "metabase",

    # Cloud / DevOps / Infra
    "terraform", "ansible", "chef", "puppet",
    "github actions", "circleci", "travis ci",
    "aws lambda", "aws rds", "aws ecs", "aws ecr",
    "aws glue", "aws athena", "aws redshift",
    "azure data factory", "azure databricks",
    "gcp pubsub", "gcp dataflow", "gcp dataproc",

    # Product / Design / UX
    "figma", "sketch", "adobe xd",
    "invision", "balsamiq",
    "user journey mapping", "service blueprinting",
    "design thinking", "wireframing", "prototyping",
    "usability testing", "user interviews", "heuristic evaluation",

    # Marketing / Growth
    "meta ads manager", "facebook ads", "instagram ads",
    "tiktok ads", "linkedin ads",
    "google tag manager", "google search console",
    "seo keyword research", "on-page seo", "technical seo",
    "crm campaigns", "lifecycle marketing",
    "marketing funnel analysis", "conversion rate optimization",
    "ab testing", "landing page optimization",

    # Email / automation
    "klaviyo", "hubspot marketing", "salesforce marketing cloud",
    "customer.io", "braze", "iterable",

    # E-commerce
    "shopify", "woocommerce", "bigcommerce", "magento",
    "product catalog management", "pricing optimization",
    "merchandising", "inventory planning",

    # Sales / customer success / RevOps
    "salesforce administration", "salesforce reporting",
    "salesforce dashboards", "salesforce flows",
    "cpq", "quote to cash",
    "salesforce service cloud", "salesforce sales cloud",
    "hubspot sales", "pipedrive", "zoho crm",
    "microsoft dynamics 365",
    "outreach", "salesloft", "apollo",
    "gong", "chorus", "zoominfo",
    "cold calling", "cold emailing",
    "account planning", "territory planning",
    "renewal management", "upsell strategy",
    "churn analysis",

    # Finance / Accounting
    "accounts payable", "accounts receivable",
    "general ledger", "reconciliation",
    "month-end close", "year-end close",
    "cash flow forecasting", "variance analysis",
    "quickbooks", "xero", "netsuite",
    "sap fico", "oracle ebs", "oracle fusion",
    "financial statement analysis",

    # Finance / Markets
    "equity valuation", "credit analysis",
    "risk modeling", "value at risk",
    "derivatives pricing", "fixed income analysis",
    "trading strategies", "portfolio optimization",
    "bloomberg terminal", "factset",

    # Data science / ML tooling
    "mlflow", "dvc", "kubeflow",
    "sagemaker", "azure ml", "vertex ai",
    "feature store", "feature engineering pipelines",

    # HR / People
    "performance management", "talent management",
    "hr analytics", "people analytics",
    "payroll processing", "time and attendance systems",

    # Operation / Logistic
    "route planning", "load optimization",
    "3pl management", "transportation management systems",

    # Healthcare
    "clinical data analysis", "healthcare analytics",
    "claims processing", "medical billing",

    # Education
    "student information systems", "learning analytics",

    # Legal / Compliance
    "contract lifecycle management", "policy compliance monitoring",

    # Creative / Media
    "brand strategy", "content creation",
    "social media management", "community management",

    # Hospitality / Retail
    "guest relations", "store operations",
    "visual merchandising", "loss prevention",

    # Construction / Engineering
    "structural analysis", "civil engineering design",

    # ITSM / IT operations
    "it service management", "incident response",
    "problem management", "change advisory board",

    # Language / Communication
    "translation", "interpretation",
    "bilingual communication", "multilingual support",
]


# Public combined master list
MASTER_SKILL_LIST = list(set(BASE_MASTER_SKILL_LIST + EXTRA_SKILLS))
