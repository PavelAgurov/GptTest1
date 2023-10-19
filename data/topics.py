"""
   Topic list and descriptions
"""
# pylint: disable=C0301

from backend.base_classes import TopicDefinition

TOPICS_LIST : list[TopicDefinition] = [
   TopicDefinition(
      1,
      "Tobacco Harm reduction",
      "Discusses the efforts of Philip Morris International (PMI) to develop and promote less harmful alternatives to traditional cigarettes. Highlights the company's commitment to finding solutions for smokers and countering misinformation.",
      url_words=["harm-reduction", "prevent-you"]
   ),
   TopicDefinition(
      2,
      "Tobacco multi-product approach",
      "Focuses on PMI's portfolio of smoke-free alternatives to traditional cigarettes, including heated tobacco products, e-vapor devices, and oral smokeless products. Discusses the company's investment in research and innovation to offer better choices for adult smokers.",
      url_words=["products"]
   ),
   TopicDefinition(
      3,
      "Inclusion, Diversity",
      "Emphasizes the importance of diversity and inclusion in the workplace, including gender equality and a range of skills. Highlights PMI's initiatives and commitment towards fostering an inclusive culture and promoting diversity.",
      url_words=["inclusion-diversity"]
   ),
   TopicDefinition(
      4,
      "Leadership content",
      "Provides insights from company leaders, including articles written by them, interviews, quotes, while also discussing decision-making, team management, and inspiration.",
      priority= 0
   ),
   TopicDefinition(
      5,
      "Investor Relations",
      "Content related to financial performance, corporate governance, and transparent communication with investors. Provides updates on results, reports, events, and essential investor resources.",
      url_words=["investor"]
   ),
   TopicDefinition(
      6,
      "Our science",
      "Highlights the scientific research and innovation behind PMI's smoke-free products. Discusses the importance of open dialogue, collaboration, and resilience in achieving a smoke-free future.",
      priority= 0,
      url_words=["intervals", "science", "our-science"]
   ),
   TopicDefinition(
      7,
      "Smoke-free vision",
      "Discusses PMI's mission to provide less harmful alternatives to traditional cigarettes and create a smoke-free world. Highlights the company's initiatives, partnerships, and efforts to raise awareness about these alternatives.",
      url_words=["smoke-free-products", "smoke-free-future"]
   ),
   TopicDefinition(
      8,
      "PMI Transformation",
      "Focuses on PMI's transformation from a traditional cigarette company to one offering smoke-free alternatives. Discusses the changes in sourcing, operations, commercialization, and revenue sources as part of this transformation.",
      url_words=["our-transformation"]
   ),
   TopicDefinition(
      9,
      "Sustainability",
      "Discusses PMI's commitment to sustainability, including reducing waste, addressing social impacts, and protecting human rights. Highlights the company's strategies, initiatives, and partnerships aimed at achieving these goals.",
      priority= 2.5,
      url_words=["sustainability"]
   ),
   TopicDefinition(
      10,
      "Regulation",
      "Regulations governing smoke-free products. Risk-based regulation and smoke-free promotion. Respect for the law. Work to prevent illegal trade. Comply with regulations. Extensive restrictions, including outright bans.",
      ["regulators", "regulation", "legal-document"],
      priority= 2.5
   ),
   TopicDefinition(
      11,
      "Jobs",
      "Highlights job opportunities and the benefits of working at PMI. Discusses the company's commitment to workplace flexibility and equal pay.",
      ["job-opportunities", "job-remotely", "job-details", "job-interview"]
   ),
   TopicDefinition(
      12,
      "Partnership and Engagement",
      "Discusses PMI's collaborative efforts, alliances, and community engagement. Highlights the company's positive impact through meaningful engagement with various external entities."
   ),
   TopicDefinition(
      13,
      "Career",
      "Career advice and discussions about career",
      ["career", "karriere", "carreras", "carrieres"],
      redirect = "Jobs"
   ),
]
