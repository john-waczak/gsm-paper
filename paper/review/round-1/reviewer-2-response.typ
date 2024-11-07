#import "response-template.typ": *

// Configure text colors for points, responses, and new text
#let (point, response, new) = configure(
  point-color: blue.darken(30%),
  response-color: black,
  new-color: green.darken(30%)
)

// Setup the rebuttal
#show: rebuttal.with(
  title: "Response to Reviewer #2 Comments",
  authors: [John Waczak and David Lary],
  date: "10-31-2024",
  paper-size: "us-letter",
)

We sincerely appreciate the time and effort that you dedicated to reviewing
our manuscript and are grateful for your insightful comments and suggested
improvements. Thank you!


#line(length: 100%)

#point[
  Please reorganize the contributions in Introduction part and highlight the
  differences with other jobs.
]<p1>

#response[
  Thank you for your comment. We have updated the introduction to explicitly
  include a list of the novel contributions made by our approach. The updated
  text now includes the following: 

  #quote[
    In summary, the key innovations introduced by the GSM are:
    - the GSM can model linear and nonlinear spectral mixing
    - the GSM does not assume the presence of pure pixels in the dataset
    - the probabilistic formulation of the GSM accounts for spectral variability
    - the simplex used for the latent space structure of the GSM is directly interpretable and forces abundances to satisfy both the abundance sum-to-one and abundance non-negativity constraints
    - the fiting procedure introduced for the GSM maintains non-negativity of endmember spectra.
  ]
]



#point[
  Give more explanations on the functions, variables, and the dimension of
  variables in equation. For example, what is  in Equation (2)? What is the
  meaning of  and  in  and  in Equation (3)?
]<p2>

#response[
  Thank you your comment. We agree that explicitly mentioning the dimension of
  included variables will improve the text. Unfortunatly, it appears that the
  symbols you've coppied into your comment are not appearing. With this in mind,
  we have made the following updates to the manuscript:

  - "$bold(x)$ (reflectance spectra)" now reads "$bold(x)$ (reflectance spectra of length $d$)"
  - The description of Equation (1) has been updated to now read #quote[
        where $bold(z) = (z_1, z_2, ..., z_(N_v) )$ corresponding to $N_v$-many
        sources and $bold(W)$ is a $D times M$ matrix of model weights which
        parameterize the mapping $psi$.
    ]
  - Equation (2) now includes the following description: #quote[
      where $delta( dot.op )$ is the Dirac delta function and $bold(z)_k$ are the
      positions of each node within the simplex.
    ]
  ]

#point[
  Authors model non-linear mixing by designing an activation function . What is
  the physical significance of this function?
]<p3>

#response[
  Thank you for your comment. Together with the model weights $bold(W)$,
  the activation function defines the mixing model, i.e. $psi(bold(z)\; bold(W)) =
  bold(W)phi(bold(z))$. We have designed these activation functions to account
  for linear mixing with $phi_(m)(bold(z_k)) = [bold(z)_k]_m$ and non-linear
  mixing via the radial basis functions described in Equation (4). To further
  clairfy this choice, we have updated the text to include the following
  statement:

  #quote[
    Equation 4 is specifically chosen so that no non-linear contributions are
    possible for pure spectra at the vertices of the simplex.
  ]
]


#point[
  Only Figure 5 is the result of compared experiments, it is suggested to
  increase the data set and do more comparative experiments.
]<p4>

#response[
  Thank you for your comment. The goal of the linear unmixing experiment
  presented in Figure 5 was to demonstrate the the GSM can successfully unmix
  _linearly_ mixed spectra without introducing unnecessary complexity, that is,
  that the GSM can successfuly solve a linear unmixing task while driving all
  non-linear terms to 0. We therefore chose NMF specifically as a well-established
  method to perform this comparrison. To make this clear, we have updated the text
  to now read as follows:

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

#point[
  The related works should be enhanced. Some recently proposed methods should be
  investigated, such as GMOGH and Rev-Net.
]<p5>

#response[
  Thank you for your comment. We have updated the introduction to refer
  to multi-objective optimization methods and have included a citation for
  GMOGH as suggested. Since this paper is explicitly concerned with non-linear
  unmixing methods, we have not cited the recent Rev-Net paper (#link("https://doi.org/10.1109/TGRS.2024.3403926"))
  which, while very interesting, appears to adopt a linear mixing model.
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


