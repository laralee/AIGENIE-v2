test_that("public AIGENIE functions expose bootEGA controls", {
  public_names <- c(
    "AIGENIE",
    "GENIE",
    "local_AIGENIE",
    "local_GENIE"
  )

  internal <- getFromNamespace(
    "run_item_reduction_pipeline",
    "AIGENIE"
  )

  expected_boot_iter <- eval(formals(internal)$boot.iter)

  for (nm in public_names) {
    fn <- getExportedValue("AIGENIE", nm)
    fml <- formals(fn)

    expect_true(
      "boot.iter" %in% names(fml),
      info = paste(nm, "must expose boot.iter")
    )
    expect_true(
      "ncores" %in% names(fml),
      info = paste(nm, "must expose ncores")
    )

    expect_identical(
      eval(fml$boot.iter),
      expected_boot_iter,
      info = paste(nm, "must preserve the internal boot.iter default")
    )

    expect_null(
      fml$ncores,
      info = paste(nm, "must default ncores to NULL")
    )

    body_text <- paste(deparse(body(fn)), collapse = "\n")

    expect_gte(
      length(gregexpr(
        "boot\\.iter\\s*=\\s*boot\\.iter",
        body_text,
        perl = TRUE
      )[[1]][gregexpr(
        "boot\\.iter\\s*=\\s*boot\\.iter",
        body_text,
        perl = TRUE
      )[[1]] > 0]),
      3L,
      label = paste(nm, "must forward boot.iter through reduction paths")
    )

    expect_gte(
      length(gregexpr(
        "ncores\\s*=\\s*ncores",
        body_text,
        perl = TRUE
      )[[1]][gregexpr(
        "ncores\\s*=\\s*ncores",
        body_text,
        perl = TRUE
      )[[1]] > 0]),
      3L,
      label = paste(nm, "must forward ncores through reduction paths")
    )
  }
})


test_that("current ncores default preserves EGAnet behavior", {
  internal <- getFromNamespace(
    "run_item_reduction_pipeline",
    "AIGENIE"
  )

  expect_null(formals(internal)$ncores)

  helper <- getFromNamespace(
    "iterative_stability_check",
    "AIGENIE"
  )

  expect_null(formals(helper)$ncores)

  helper_body <- paste(deparse(body(helper)), collapse = "\n")

  expect_match(
    helper_body,
    "if \\(!is\\.null\\(ncores\\)\\)",
    perl = TRUE
  )

  expect_match(
    helper_body,
    "boot_args\\$ncores\\s*<-\\s*ncores",
    perl = TRUE
  )
})
