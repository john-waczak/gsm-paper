#import "response-template.typ": *

// Configure text colors for points, responses, and new text
#let (point, response, new) = configure(
  point-color: blue.darken(30%),
  response-color: black,
  new-color: green.darken(30%)
)

// Setup the rebuttal
#show: rebuttal.with(
  title: "Response to Reviewer #1 Comments",
  authors: [John Waczak and David Lary],
  date: "10-31-2024",
  paper-size: "us-letter",
)

We sincerely appreciate the time and effort that you dedicated to reviewing
our manuscript and are grateful for your insightful comments and suggested
improvements. Thank you!


#line(length: 100%)

#point[
    The gridded simplex framework proposed in the manuscript is utilized to
    replace the rectangular framework in GTM. However, both frameworks are grounded
    on the assumption that the data is uniformly sampled from the embedded manifold
    in the data space. This assumption may not hold true in real-world scenarios.
    How do the authors address this potential issue?
]<p1>

#response[
  Thank you for your comment. We agree that the assumption of uniform sampling
  in the latent space is a key limitation of the original GTM formulation. For
  the GSM we are introducing here, we have addressed this by introducing adaptive
  mixing coefficients $pi_k$ which are updated during the fitting procedure. As
  a consequence, the prior probability distriubtion is no longer uniform across
  nodes in the GSM latent space. Previously, we wrote "The second change is to
  replace equal prior probabilities with adaptive mixing coefficients $pi_k$ to
  allow the GSM to model HSI with nonuniform mixing distributions". To further
  clarify this point, we have revised the text to now read: 

    #quote[
    The second change is to augment the equal prior probabilities of the GTM
    latent space with adaptive mixing coefficients $pi_k$ to allow for nonuniform
    sampling across the abundance simplex. This addition allows the GSM flexibly
    address many possible mixing scenarios without an _a priori_ assumption for a
    specific mixing distribution.
  ]

]


#point[
    The adaptive mixing coefficient is used instead of the equal prior probability
    of GTM to achieve the explanation of the endmember variation phenomenon.
    However, the adaptive update of the accuracy parameter β actually leads to
    the endmember variability estimated by the model being a range related to β
    in each band. How does the author determine the final endmember spectrum with
    variability?
]<p2>

#response[
    Thank you for your comment. The adaptive mixing coefficients $pi_k$ are
    utilized to account for nonuniform sampling of possible mixtures, not for
    estimating spectral variablity. As you correctly point out, we use the precision
    parameter $beta$ for this corresponding to a normal distribution with a
    standard deviation of $sqrt(beta^(-1))$for all wavelengths. In future
    work we plan to extend the model to allow for wavelength-dependent spectral
    variability.
]



#point[
    In the experimental section of the manuscript,  the proposed method was
    compared with three early NMF methods. However, the reviewer believes that the
    experimental evaluation was not comprehensive and sufficient. Therefore, it is
    recommended that the authors compare their work with some of the latest and most
    advanced methods, such as nonlinear unmixing and spectral variability methods,
    as this, would enhance the contribution of the work in this paper.
]<p3>

#response[
    Thank you for your comment. The goal of our first experiment was to
    demonstrate that the GSM can model _linearly mixed_ data without introducing
    unnecessary complexity, or in other words, the GSM should not incorporate non-linear
    effects for linearly mixed data. Our intention was not to suggest that the
    GSM is superior to other linear mixing models but rather provide a fair,
    apples-to-apples comparison to an established approach. To clarify this point,
    we have revised the text introducing the experiment to now read as follows:

    #quote[   
      To illustrate the effectiveness of the GSM, we first demonstrate its ability to
      model linear mixing. This serves as an important limiting case since linearly
      mixed spectra should not lead to the spurous introduction of non-linear
      contributions by the GSM. The goal of this first test is therefore to
      demonstrate that the GSM drives non-linear weights to zero for linearly mixed
      data while providing a fair test to compare the GSM to a well-established linear
      mixing model. This ability clealy distinguishes the GSM from other non-nonlinear
      unmixing approaches such as autoencoders, which by their design, include
      non-linear mixing even when it is not present in the underlying data.
    ]

    Additionaly, we have included the following line after describing the
    varieties of NMF we chose to use:

    #quote[
      We note that the goal of this test is not to prove the GSM is superior to
      other models for linear mixing, but rather to demonstrate that the GSM can
      model linearly mixed data without introducing unnecessary complexity.
    ]
]




// #reviewer()
// This reviewers' feedback was...

// #point[
//   There appears to be an error...
// ]<p1>

// #response[
//   #lorem(20).

//   The revised text now reads:
//   #quote[
//     #lorem(10) #new[#lorem(2)].
//   ]
// ]

// #point[
//   #lorem(10).
// ]

// #response[
//   See response to @pt-p1.
//   Similar to the `i-figured` package, references to labeled `point`s must be prefixed by `pt-` as in `@pt-p1` which refers to the `point` labeled `<p1>`.
// ]


