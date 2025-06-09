# Medicare_Matchmaker_Team5A
Medicare Matchmaker - DAIS 2025 Hackathon

# Overview
Matching Medicare coverage with applicable providers given patient medical concerns, their location, and their budget. 

## Inspiration
Navigating health insurance options can be overwhelming, especially when trying to match plans to specific medical needs, locations, and budgets. Our goal with Medicare Matchmaker was to empower users with clear, personalized insights into their coverage options—based on what matters most to them.

## What it does
Medicare Matchmaker is an intelligent agent that helps users explore medical coverage options tailored to their location, copay preferences, and healthcare needs. Users can ask free-form questions, and the agent returns plans available in their area—prioritizing those with $0 copays for the specified services.

## How we built it
We utilized the mimilabs mpf_benefit_summary and landscape_medicare_advantage datasets to gather detailed information on plans and service-specific copays by location.
Key steps:

- Consolidated relevant data into a streamlined queryable view.

- Built an agent using LangChain to parse natural language input and extract key parameters like service type, location, and copay thresholds.

- Queried our dataset with these filters to return the most relevant plans, prioritizing affordability and accessibility.

- Used Nimble API to integrate service provider quality based on Google reviews 

## Challenges we ran into
- We aimed to expand our tool to include local healthcare providers and their accepted plans, but we couldn’t find a unified dataset linking doctors to coverage regions.

## Accomplishments that we're proud of
- Created a natural language interface that accepts multiple filters in a single query.

- Delivered tailored insurance plan recommendations using real-world data.

- Helped demystify complex healthcare choices in a user-friendly format.


## What we learned
- How to integrate multiple healthcare datasets using Delta Shares.

- How to set up multi-agent architectures and tools with LangChain.

- Best practices for interpreting user input and returning relevant, ranked results.

- How to use a Nimble API call with an Agent

## What's next for Medicare Matchmaker
- Recommending top-reviewed local providers based on user location and insurance compatibility.

- Expanding to include Medicaid and private insurance plans for broader accessibility.
