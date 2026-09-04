"""Standard evaluation question set for the RAG benchmark.

Each item:
  query:            the user question
  expected_source:  which knowledge doc should be retrieved
  reference:        a model answer used for ROUGE comparison
"""

BENCHMARK_QUESTIONS = [
    {
        "query": "What does dwelling coverage in homeowners insurance pay for?",
        "expected_source": "homeowners_insurance.txt",
        "relevant_sources": ["homeowners_insurance.txt"],
        "reference": "Dwelling coverage pays to repair or rebuild your home if damaged by a covered peril, including the structure and attached structures.",
    },
    {
        "query": "What is the difference between collision and comprehensive auto coverage?",
        "expected_source": "auto_insurance.txt",
        "relevant_sources": ["auto_insurance.txt"],
        "reference": "Collision covers damage from a collision with another vehicle or object. Comprehensive covers non-collision events like theft, fire, and vandalism.",
    },
    {
        "query": "What is cash value in life insurance?",
        "expected_source": "life_insurance.txt",
        "relevant_sources": ["life_insurance.txt"],
        "reference": "Cash value is the savings component of a permanent life insurance policy that grows on a tax-deferred basis and can be borrowed against.",
    },
    {
        "query": "What does general liability insurance cover?",
        "expected_source": "business_liability.txt",
        "relevant_sources": ["business_liability.txt"],
        "reference": "General liability insurance covers bodily injury, property damage, medical payments, and personal or advertising injury claims.",
    },
    {
        "query": "What is First Notice of Loss in the claims process?",
        "expected_source": "claims_handling.txt",
        "relevant_sources": ["claims_handling.txt"],
        "reference": "First Notice of Loss is when the policyholder reports the loss to the insurer, capturing policy number, incident details, date, location, and parties.",
    },
    {
        "query": "How does term life insurance differ from whole life insurance?",
        "expected_source": "life_insurance.txt",
        "relevant_sources": ["life_insurance.txt"],
        "reference": "Term life provides coverage for a specified period without cash value. Whole life provides lifelong coverage with a cash value component.",
    },
    {
        "query": "What does medical payments coverage in homeowners insurance pay for?",
        "expected_source": "homeowners_insurance.txt",
        "relevant_sources": ["homeowners_insurance.txt"],
        "reference": "Medical payments coverage pays for medical expenses for guests injured on your property, regardless of fault.",
    },
    {
        "query": "What protection does uninsured motorist coverage provide?",
        "expected_source": "auto_insurance.txt",
        "relevant_sources": ["auto_insurance.txt"],
        "reference": "Uninsured motorist coverage protects you if hit by a driver with no insurance or insufficient coverage.",
    },
    {
        "query": "What is subrogation in claims handling?",
        "expected_source": "claims_handling.txt",
        "relevant_sources": ["claims_handling.txt"],
        "reference": "Subrogation is recovering claim payments from the responsible party's insurer when a third party is liable.",
    },
    {
        "query": "What does cyber liability insurance cover?",
        "expected_source": "business_liability.txt",
        "relevant_sources": ["business_liability.txt"],
        "reference": "Cyber liability coverage helps businesses respond to data breaches and cyberattacks, covering notification costs, legal fees, regulatory fines, and investigations.",
    },
]
