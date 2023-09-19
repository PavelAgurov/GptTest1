"""
   Topic list and descriptions
"""
# pylint: disable=C0301

from backend.base_classes import TopicDefinition

TOPICS_LIST : list[TopicDefinition] = [
   TopicDefinition(
      1,
      "Tobacco Harm reduction",
      "Focuses on harm reduction applied to tobacco and exploring the scientific basis behind it, highlighting the development of smoke-free alternatives to traditional cigarettes supported by research and technology."
   ),
   TopicDefinition(
      2,
      "Tobacco multi-product approach",
      "Several categories of smokeless alternatives (e-vaping devices, heated tobacco products, oral smokeless products and other)."
   ),
   TopicDefinition(
      3,
      "Inclusion, Diversity",
      "Highlighting the significance of fostering an inclusive workplace through initiatives, training, practices, and policies, while also integrating content featuring diversity metrics, successful case studies, and resources for cultivating diversity and inclusivity in organizational culture."
      ),
   TopicDefinition(
      4,
      "Leadership content",
      "Centers on leadership within company, covering leadership principles, strategies, and insights from thought leaders, including articles, interviews, and perspectives, while also discussing effective communication, decision-making, team management, and inspiration."
   ),
   TopicDefinition(
      5,
      "Investor Relations",
      "Your comprehensive resource for financial performance, corporate governance, and transparent communication with investors, providing updates on results, reports, events, and essential investor resources."
   ),
   TopicDefinition(
      6,
      "Our science",
      "The scientific foundation of smoke-free vision, showcasing research, innovations, methodologies, technologies, and breakthroughs across various fields, including collaborations, experiments, publications, and applications.",
      ["intervals", "science"]
   ),
   TopicDefinition(
      7,
      "Smoke-free vision",
      "Centers on societal and corporate mission, highlighting its dedication to providing millions of smokers with less harmful but satisfying alternatives through innovative smoke-free products, aiming to accelerate the transition towards a cigarette-free world."
   ),
   TopicDefinition(
      8,
      "PMI Transformation",
      "Centers on transformation from a cigarette-focused company to one offering improved smoke-free alternatives, marking a significant historical shift with organizational, management, and vision implications, aiming to drive positive societal change towards a smoke-free future."
   ),
   TopicDefinition(
      9,
      "Sustainability",
#      "Exploring Environmental, Social, and Governance principles for a greener, more responsible future, encompassing diverse subjects like renewable energy, waste reduction, ethical sourcing, and sustainable development goals."
      "Any aspects of corporate sustainability initiatives including sustainability in supply chain and more responsible future."
   ),
   TopicDefinition(
      10,
      "Regulation",
#      "Focuses on the need for a regulatory framework to encourage smoke-free products, highlighting how governments and public health bodies can regulate these alternatives to promote public health and reduce cigarette sales for a smoke-free future.",
     "Explores strict regulations and enforcement on tobacco products, emphasizing the importance of innovation in smoke-free alternatives to address smoking-related harm while protecting public health.",
     ["regulators"]
   ),
   TopicDefinition(
      11,
      "Jobs",
      "Your gateway to explore career opportunities, offering valuable insights into job roles, qualifications, and the recruitment process, along with current job openings, application details, and the benefits of joining our organization.",
      ["job-opportunities", "job-remotely", "job-details", "job-interview"]
   ),
   TopicDefinition(
      12,
      "Partnership and Engagement",
      "Collaborative efforts, strategic alliances, and community engagement fostering positive impact through meaningful engagement with various external entities."
   ),
   TopicDefinition(
      13,
      "Top management content",
      "Experiences and perspectives shared by senior management or top executives of a company, including their roles, strategies, and the impact of their decisions on the company."
   )
]
