Export path using:-
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"


PHASE - 2

`
expand
    ↓
research (parallel)
    ↓
critic
    → if bad → research retry
    → if good → fact_check
               → if ok → synthesize
               → if needs more info → research retry
                         ↓
                  contradiction_check
                         ↓
                      synthesize
`