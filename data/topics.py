"""
   Topic list and descriptions
"""
# pylint: disable=C0301

from backend.base_classes import TopicDefinition

TOPICS_LIST : list[TopicDefinition] = [
   TopicDefinition(
      1,
      "Tobacco Harm reduction",
      "Discusses the development and promotion of smoke-free alternatives to traditional cigarettes, such as heated tobacco products, e-vapor devices, and oral smokeless products, as a strategy to reduce the harmful effects of smoking. Highlights the scientific research and investment behind these products and their potential impact on public health.",
      url_words=["harm-reduction"]
   ),
   TopicDefinition(
      2,
      "Tobacco multi-product approach",
      "Focuses on PMI's diverse portfolio of smoke-free alternatives to traditional cigarettes, including heated tobacco products, e-vapor devices, and oral smokeless products. Discusses the development, commercialization, and global distribution of these products, as well as their acceptance by adult smokers.",
      url_words=["products"]
   ),
   TopicDefinition(
      3,
      "Inclusion, Diversity",
      "Emphasizes the importance of diversity and inclusion in the workplace, including gender equality and a range of skills. Discusses initiatives, practices, and resources aimed at fostering an inclusive culture and promoting diversity. Also highlights the role of diversity in driving innovation and the company's transformation towards a smoke-free future.",
      url_words=["inclusion-diversity"]
   ),
   TopicDefinition(
      4,
      "Leadership content",
      "Shares insights from leaders on the importance of agility, moral compass, and teamwork in driving change and achieving the company's smoke-free vision. Discusses the role of gender diversity in STEM fields and the need for leaders to challenge unconscious bias and learn from younger talent.",
      priority= 2
   ),
   TopicDefinition(
      5,
      "Investor Relations",
      "Content related to financial performance, corporate governance, and transparent communication with investors, providing updates on results, reports, events, and essential investor resources.",
      url_words=["investor-relations", "inverstor"]
   ),
   TopicDefinition(
      6,
      "Our science",
      "Highlights the scientific research and innovation behind PMI's smoke-free vision and products. Discusses the importance of open dialogue, collaboration, and resilience in addressing global challenges and achieving a smoke-free future.",
      ["intervals", "science", "our-science"]
   ),
   TopicDefinition(
      7,
      "Smoke-free vision",
      "Discusses PMI's mission to provide less harmful alternatives to traditional cigarettes and create a smoke-free world. Highlights the company's initiatives, partnerships, and efforts to raise awareness about these alternatives and counter misinformation.",
      url_words=["smoke-free-products"]
   ),
   TopicDefinition(
      8,
      "PMI Transformation",
      "Focuses on PMI's transformation from a traditional cigarette company to one offering smoke-free alternatives. Discusses the changes in sourcing, operations, commercialization, and revenue sources as part of this transformation. Also highlights the company's expansion into wellness and healthcare products.",
      url_words=["our-transformation"]
   ),
   TopicDefinition(
      9,
      "Sustainability",
      "Discusses PMI's commitment to sustainability, including reducing post-consumer waste, addressing social impacts in its supply chain, and protecting human rights. Highlights the company's strategies, initiatives, and partnerships aimed at achieving these goals.",
      priority= 2,
      url_words=['sustainability']
   ),
   TopicDefinition(
      10,
      "Regulation",
      "Discusses the regulations governing smoke-free products and PMI's marketing practices. Highlights the company's stance on these regulations and its advocacy for sensible, risk-based regulation of smoke-free products.",
      ["regulators", "regulation"],
      priority= 2
   ),
   TopicDefinition(
      11,
      "Jobs",
      "Highlights job opportunities at PMI as part of the company's transformation towards a smoke-free future. Discusses the roles, responsibilities, and benefits of working at PMI, as well as the company's commitment to workplace flexibility and equal pay.",
      ["job-opportunities", "job-remotely", "job-details", "job-interview", "career"]
   ),
   TopicDefinition(
      12,
      "Partnership and Engagement",
      "Collaborative efforts, strategic alliances, and community engagement fostering positive impact through meaningful engagement with various external entities."
   ),
   # TopicDefinition(
   #    13,
   #    "Leadership content",
   #    "Any content that mentions at least twice senior management or top managers of PMI (including SEO, COO, SVP, VP etc.) OR content created by them (as indicated by quotes or the use of a first person point of view).",
   #    priority= 2,
   #    url_words=["leaders"]
   # )
]
