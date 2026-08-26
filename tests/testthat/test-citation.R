test_that("AIGENIE exposes software and methodology citations", {
  cites <- utils::citation("AIGENIE")

  expect_gte(length(cites), 2L)

  printed <- paste(
    capture.output(print(cites, bibtex = TRUE)),
    collapse = "\n"
  )

  expect_true(
    grepl(
      "10.32614/CRAN.package.AIGENIE",
      printed,
      fixed = TRUE
    )
  )

  expect_true(
    grepl(
      "10.3758/s13428-026-03082-1",
      printed,
      fixed = TRUE
    )
  )
})
