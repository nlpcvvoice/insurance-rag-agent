"""Standard evaluation question set for the RAG benchmark.

Each item:
  query:            the user question
  expected_source:  which knowledge doc should be retrieved
  relevant_sources: docs that contain answer-relevant content (can be more than one)
  reference:        a model answer used for comparison

Expanded 2026-09-04 from 10 to 50 questions across 15 corpus documents
(5 original + 10 new). Includes overlapping/confusable topics (liability family,
income-replacement family) so retrieval is not trivially saturated.
"""

BENCHMARK_QUESTIONS = [
    # ---- homeowners_insurance.txt (1-6) ----
    {
        "query": "What does dwelling coverage in homeowners insurance pay for?",
        "expected_source": "homeowners_insurance.txt",
        "relevant_sources": ["homeowners_insurance.txt"],
        "reference": "Dwelling coverage pays to repair or rebuild your home if damaged by a covered peril, including the structure, attached structures, and built-in appliances.",
    },
    {
        "query": "What does medical payments coverage in homeowners insurance pay for?",
        "expected_source": "homeowners_insurance.txt",
        "relevant_sources": ["homeowners_insurance.txt", "renters_insurance.txt"],
        "reference": "Medical payments coverage pays for medical expenses for guests injured on your property, regardless of fault.",
    },
    {
        "query": "Which perils do standard homeowners policies exclude?",
        "expected_source": "homeowners_insurance.txt",
        "relevant_sources": ["homeowners_insurance.txt", "flood_insurance.txt"],
        "reference": "Standard homeowners policies exclude flooding, earthquakes, nuclear accidents, war, intentional damage, and damage from lack of maintenance.",
    },
    {
        "query": "What does loss of use coverage for homeowners pay when a home is uninhabitable?",
        "expected_source": "homeowners_insurance.txt",
        "relevant_sources": ["homeowners_insurance.txt"],
        "reference": "Loss of use coverage pays additional living expenses such as hotel stays and meals while the home is repaired after a covered loss.",
    },
    {
        "query": "What is the difference between Coverage A dwelling and Coverage B other structures?",
        "expected_source": "homeowners_insurance.txt",
        "relevant_sources": ["homeowners_insurance.txt"],
        "reference": "Coverage A insures the dwelling structure and attached structures, while Coverage B covers detached structures on the property such as garages, sheds, and fences.",
    },
    {
        "query": "Does a homeowners policy cover a detached garage on the property?",
        "expected_source": "homeowners_insurance.txt",
        "relevant_sources": ["homeowners_insurance.txt"],
        "reference": "Yes, other structures coverage pays for structures not attached to the home, including detached garages, sheds, and fences.",
    },
    # ---- renters_insurance.txt (7-10) ----
    {
        "query": "What does renters insurance cover for a tenant living in an apartment?",
        "expected_source": "renters_insurance.txt",
        "relevant_sources": ["renters_insurance.txt", "flood_insurance.txt"],
        "reference": "Renters insurance covers personal property, loss of use, personal liability, and medical payments for tenants, but not the building structure itself.",
    },
    {
        "query": "Why do landlords require renters insurance in a lease?",
        "expected_source": "renters_insurance.txt",
        "relevant_sources": ["renters_insurance.txt"],
        "reference": "Landlords require it to shield their own property policy from tenant liability claims and to reimburse their policy for accidental tenant-caused damage.",
    },
    {
        "query": "Is a roommate automatically covered under another tenant's renters policy?",
        "expected_source": "renters_insurance.txt",
        "relevant_sources": ["renters_insurance.txt"],
        "reference": "No, roommates not named on the policy are generally not covered and each roommate needs their own renters policy.",
    },
    {
        "query": "Does renters insurance cover flood damage to a tenant's belongings?",
        "expected_source": "renters_insurance.txt",
        "relevant_sources": ["renters_insurance.txt", "flood_insurance.txt"],
        "reference": "No, renters policies exclude flooding, so a separate flood policy is needed for flood damage to belongings.",
    },
    # ---- auto_insurance.txt (11-15) ----
    {
        "query": "What is the difference between collision and comprehensive auto coverage?",
        "expected_source": "auto_insurance.txt",
        "relevant_sources": ["auto_insurance.txt", "commercial_auto_insurance.txt"],
        "reference": "Collision covers damage from a collision with another vehicle or object. Comprehensive covers non-collision events like theft, fire, storm damage, and vandalism.",
    },
    {
        "query": "What protection does uninsured motorist coverage provide?",
        "expected_source": "auto_insurance.txt",
        "relevant_sources": ["auto_insurance.txt", "commercial_auto_insurance.txt"],
        "reference": "Uninsured motorist coverage protects you if you are hit by a driver with no insurance or with insufficient coverage for the damages.",
    },
    {
        "query": "If I use my personal car for business deliveries, does my personal auto policy cover me?",
        "expected_source": "commercial_auto_insurance.txt",
        "relevant_sources": ["commercial_auto_insurance.txt", "auto_insurance.txt"],
        "reference": "Personal auto policies typically exclude business use, so a commercial auto policy is needed for vehicles used for deliveries or business purposes.",
    },
    {
        "query": "What does commercial auto physical damage coverage include beyond liability?",
        "expected_source": "commercial_auto_insurance.txt",
        "relevant_sources": ["commercial_auto_insurance.txt"],
        "reference": "Commercial auto physical damage coverage includes comprehensive and collision coverages for the business vehicle, subject to a deductible.",
    },
    {
        "query": "When would a business be issued a fleet commercial auto policy?",
        "expected_source": "commercial_auto_insurance.txt",
        "relevant_sources": ["commercial_auto_insurance.txt"],
        "reference": "Businesses with several vehicles choose a fleet policy, which schedules multiple vehicles under one agreement and streamlines claims and premium management.",
    },
    # ---- life_insurance.txt (16-19) ----
    {
        "query": "What is cash value in life insurance?",
        "expected_source": "life_insurance.txt",
        "relevant_sources": ["life_insurance.txt", "annuities.txt"],
        "reference": "Cash value is the savings component of a permanent life insurance policy that grows on a tax-deferred basis and can be borrowed against.",
    },
    {
        "query": "How does term life insurance differ from whole life insurance?",
        "expected_source": "life_insurance.txt",
        "relevant_sources": ["life_insurance.txt"],
        "reference": "Term life provides coverage for a specified period without cash value. Whole life provides lifelong coverage with a cash value component.",
    },
    {
        "query": "What is a beneficiary in a life insurance policy?",
        "expected_source": "life_insurance.txt",
        "relevant_sources": ["life_insurance.txt"],
        "reference": "A beneficiary is the person or entity who receives the death benefit when the insured passes away.",
    },
    {
        "query": "Does a term life policy accumulate cash value?",
        "expected_source": "life_insurance.txt",
        "relevant_sources": ["life_insurance.txt", "annuities.txt"],
        "reference": "No, term life insurance is pure protection for a set period and does not accumulate cash value, unlike permanent life policies.",
    },
    # ---- annuities.txt (20-22) ----
    {
        "query": "What is an annuity and what is it designed for?",
        "expected_source": "annuities.txt",
        "relevant_sources": ["annuities.txt"],
        "reference": "An annuity is an insurance-company financial product that provides income payments over time, typically designed for retirement income.",
    },
    {
        "query": "What is the difference between a fixed and a variable annuity?",
        "expected_source": "annuities.txt",
        "relevant_sources": ["annuities.txt", "life_insurance.txt"],
        "reference": "A fixed annuity guarantees a rate of return and payout, while a variable annuity invests in subaccounts and its income fluctuates with market performance.",
    },
    {
        "query": "What is a surrender charge on an annuity?",
        "expected_source": "annuities.txt",
        "relevant_sources": ["annuities.txt"],
        "reference": "A surrender charge is a fee for withdrawing annuity funds early, typically declining over the first several contract years.",
    },
    # ---- umbrella_insurance.txt (23-25) ----
    {
        "query": "What is personal umbrella insurance?",
        "expected_source": "umbrella_insurance.txt",
        "relevant_sources": ["umbrella_insurance.txt", "business_liability.txt"],
        "reference": "Personal umbrella insurance provides extra liability coverage above the limits of your underlying homeowners, auto, and watercraft policies.",
    },
    {
        "query": "When does an umbrella policy begin to pay?",
        "expected_source": "umbrella_insurance.txt",
        "relevant_sources": ["umbrella_insurance.txt"],
        "reference": "An umbrella policy pays only after the underlying policy limits are exhausted and begins its excess layer above existing limits.",
    },
    {
        "query": "Who should typically consider buying umbrella insurance?",
        "expected_source": "umbrella_insurance.txt",
        "relevant_sources": ["umbrella_insurance.txt", "homeowners_insurance.txt"],
        "reference": "People with significant assets, homeowners with pools or trampolines, parents of teenage drivers, landlords, and frequent event hosts should consider it.",
    },
    # ---- business_liability.txt (26-28) ----
    {
        "query": "What does general liability insurance cover?",
        "expected_source": "business_liability.txt",
        "relevant_sources": ["business_liability.txt", "professional_liability.txt"],
        "reference": "General liability insurance covers bodily injury, property damage, medical payments, and personal or advertising injury claims against a business.",
    },
    {
        "query": "What does cyber liability insurance cover?",
        "expected_source": "business_liability.txt",
        "relevant_sources": ["business_liability.txt"],
        "reference": "Cyber liability coverage helps businesses respond to data breaches and cyberattacks, covering notification costs, legal fees, regulatory fines, and investigations.",
    },
    {
        "query": "Which type of liability policy protects a business against damages from a physical slip-and-fall on its premises?",
        "expected_source": "business_liability.txt",
        "relevant_sources": ["business_liability.txt", "professional_liability.txt"],
        "reference": "General liability insurance covers bodily injury such as a slip-and-fall on the business premises, including medical payments and defense.",
    },
    # ---- professional_liability.txt (29-33) ----
    {
        "query": "What is errors and omissions insurance for a professional?",
        "expected_source": "professional_liability.txt",
        "relevant_sources": ["professional_liability.txt", "business_liability.txt"],
        "reference": "E&O coverage protects professionals against claims of negligence, mistakes, or failure to perform professional duties, including defense and settlements.",
    },
    {
        "query": "How does claims-made E&O coverage differ from an occurrence policy?",
        "expected_source": "professional_liability.txt",
        "relevant_sources": ["professional_liability.txt"],
        "reference": "A claims-made policy covers claims reported while the policy is active, whereas an occurrence policy responds to events during the policy period regardless of when the claim is filed.",
    },
    {
        "query": "What does tail coverage do for an E&O policy?",
        "expected_source": "professional_liability.txt",
        "relevant_sources": ["professional_liability.txt"],
        "reference": "Tail coverage extends protection for work done before the policy is canceled, covering claims reported after cancellation for prior work.",
    },
    {
        "query": "A financial advisor is sued for bad investment advice. Which coverage responds?",
        "expected_source": "professional_liability.txt",
        "relevant_sources": ["professional_liability.txt", "business_liability.txt"],
        "reference": "Professional liability (E&O) insurance responds to claims of negligent professional advice or errors, providing defense and settlement coverage.",
    },
    {
        "query": "Which professionals typically carry errors and omissions insurance?",
        "expected_source": "professional_liability.txt",
        "relevant_sources": ["professional_liability.txt"],
        "reference": "Consultants, accountants, attorneys, architects, engineers, IT firms, real estate agents, and financial advisors typically carry E&O coverage.",
    },
    # ---- workers_compensation.txt (34-38) ----
    {
        "query": "What does workers compensation insurance provide to injured employees?",
        "expected_source": "workers_compensation.txt",
        "relevant_sources": ["workers_compensation.txt", "disability_insurance.txt"],
        "reference": "Workers compensation provides wage replacement, medical benefits, and death benefits to employees for work-related injuries or illnesses.",
    },
    {
        "query": "How are workers compensation premiums calculated?",
        "expected_source": "workers_compensation.txt",
        "relevant_sources": ["workers_compensation.txt"],
        "reference": "Premiums are based on industry classification, payroll, and claims history, with a poor experience modification factor raising the premium.",
    },
    {
        "query": "What injuries are typically excluded from workers compensation coverage?",
        "expected_source": "workers_compensation.txt",
        "relevant_sources": ["workers_compensation.txt"],
        "reference": "Injuries from intoxication, self-inflicted harm, willful misconduct, fighting at work, and off-the-job injuries are typically excluded.",
    },
    {
        "query": "Are independent contractors covered under an employer's workers compensation policy?",
        "expected_source": "workers_compensation.txt",
        "relevant_sources": ["workers_compensation.txt", "professional_liability.txt"],
        "reference": "No, independent contractors are generally not covered under an employer's workers compensation policy.",
    },
    {
        "query": "What death benefits does workers compensation pay?",
        "expected_source": "workers_compensation.txt",
        "relevant_sources": ["workers_compensation.txt"],
        "reference": "Death benefits pay funeral expenses and a percentage of wages to the dependents of a worker who dies from a work-related cause.",
    },
    # ---- disability_insurance.txt (39-42) ----
    {
        "query": "What is the difference between short-term and long-term disability insurance?",
        "expected_source": "disability_insurance.txt",
        "relevant_sources": ["disability_insurance.txt", "workers_compensation.txt"],
        "reference": "Short-term disability pays for a few weeks to a few months after a short elimination period, while long-term disability covers extended periods with a longer elimination period.",
    },
    {
        "query": "What is the difference between own-occupation and any-occupation disability definitions?",
        "expected_source": "disability_insurance.txt",
        "relevant_sources": ["disability_insurance.txt"],
        "reference": "Own-occupation pays if you cannot perform your own occupation, while any-occupation pays only if you cannot work in any occupation you are reasonably suited for.",
    },
    {
        "query": "How does the elimination period affect a disability policy premium?",
        "expected_source": "disability_insurance.txt",
        "relevant_sources": ["disability_insurance.txt"],
        "reference": "Longer elimination periods lower the premium because they push the benefit start later and reduce the insurer's exposure.",
    },
    {
        "query": "Are disability insurance benefits taxable if the employer paid the premiums?",
        "expected_source": "disability_insurance.txt",
        "relevant_sources": ["disability_insurance.txt"],
        "reference": "Benefits are usually taxable as income if the employer paid the premium, and generally tax-free if the premium was paid with after-tax dollars.",
    },
    # ---- commercial_property.txt (43-46) ----
    {
        "query": "What is business income coverage in commercial property insurance?",
        "expected_source": "commercial_property.txt",
        "relevant_sources": ["commercial_property.txt"],
        "reference": "Business income coverage replaces lost income and continues selected operating expenses when a covered loss forces the business to suspend operations.",
    },
    {
        "query": "What is the difference between replacement cost and actual cash value property valuation?",
        "expected_source": "commercial_property.txt",
        "relevant_sources": ["commercial_property.txt"],
        "reference": "Replacement cost pays to replace an item with a similar new item, while actual cash value subtracts depreciation from the replacement cost.",
    },
    {
        "query": "A restaurant loses revenue for two months after a fire. Which commercial coverage helps pay those ongoing costs?",
        "expected_source": "commercial_property.txt",
        "relevant_sources": ["commercial_property.txt"],
        "reference": "Business income (business interruption) coverage replaces lost income and continues operating expenses during the suspension of operations.",
    },
    {
        "query": "Does commercial property insurance cover flood damage to a business?",
        "expected_source": "commercial_property.txt",
        "relevant_sources": ["commercial_property.txt", "flood_insurance.txt"],
        "reference": "No, commercial property policies exclude flooding unless a flood endorsement or separate flood policy is added.",
    },
    # ---- flood_insurance.txt (47-49) ----
    {
        "query": "Why is flood coverage excluded from standard home and renters policies?",
        "expected_source": "flood_insurance.txt",
        "relevant_sources": ["flood_insurance.txt", "homeowners_insurance.txt"],
        "reference": "Flood risk is geographically concentrated and potentially catastrophic, so private insurers exclude it and a separate flood policy is required.",
    },
    {
        "query": "What is the typical waiting period before flood insurance becomes effective?",
        "expected_source": "flood_insurance.txt",
        "relevant_sources": ["flood_insurance.txt"],
        "reference": "There is commonly a 30-day waiting period after purchase before flood coverage becomes effective, so it cannot be bought at the last minute before an event.",
    },
    {
        "query": "What factors determine a flood insurance premium?",
        "expected_source": "flood_insurance.txt",
        "relevant_sources": ["flood_insurance.txt"],
        "reference": "Premiums depend on flood zone, elevation relative to base flood elevation, structure type, and coverage amounts.",
    },
    # ---- travel_insurance.txt (50-52) ----
    {
        "query": "What does trip cancellation insurance reimburse under a travel policy?",
        "expected_source": "travel_insurance.txt",
        "relevant_sources": ["travel_insurance.txt"],
        "reference": "Trip cancellation insurance reimburses prepaid, non-refundable trip costs if you must cancel for a covered reason such as illness, injury, or death of the traveler or family member.",
    },
    {
        "query": "What does emergency medical and evacuation coverage in travel insurance pay for?",
        "expected_source": "travel_insurance.txt",
        "relevant_sources": ["travel_insurance.txt"],
        "reference": "Emergency medical coverage pays for hospital care, doctors, and prescriptions abroad, while medical evacuation coverage transports you to an adequate facility or home when medically necessary.",
    },
    {
        "query": "Why do pre-existing medical conditions often require a waiver in travel insurance?",
        "expected_source": "travel_insurance.txt",
        "relevant_sources": ["travel_insurance.txt"],
        "reference": "Travel policies exclude pre-existing medical conditions unless a waiver is purchased, usually requiring the policy be bought within a short period after the first trip deposit.",
    },
]