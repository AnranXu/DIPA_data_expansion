/*import { Stack, Typography } from "@mui/material";
import React from "react";
import InformativenessStar from "./star.js";

// props
// value: number (-3 <= n <=3)

const getText = (value) => {
    let firstWord = "";
    let secondWord = "";
    if (value < 0) {
        secondWord = "Uninformative";
    } else if (value === 0) {
        secondWord = "Neutral";
    } else if (value > 0) {
        secondWord = "Informative";
    }

    const abs = Math.abs(value);
    if (abs === 3) {
        firstWord = "Extremely";
    } else if (abs === 2) {
        firstWord = "Moderately";
    } else if (abs === 1) {
        firstWord = "Slightly";
    }

    return firstWord + " " + secondWord;
};

const Informativeness = (props) => {
    const values = [-3, -2, -1, 0, 1, 2, 3];
    console.log(props.value);

    return (
        <Stack>
            <Typography variant="h5" fontWeight="bold">
                Informativeness
            </Typography>
            <Stack alignItems="center" paddingTop="10px">
                <Stack
                    direction="row"
                    spacing="10px"
                    alignItems="center"
                    justifyContent="center"
                >
                    {values.map((value, i) => {
                        return (
                            <InformativenessStar
                                key={i}
                                filled={value <= props.value}
                            />
                        );
                    })}
                </Stack>
                <Typography variant="h6" fontWeight="normal">
                    {getText(props.value)}
                </Typography>
            </Stack>
        </Stack>
    );
    // return (
    //     <Stack>
    //         <Typography variant="h5" fontWeight="bold">
    //             Informativeness
    //         </Typography>
    //         <Stack alignItems="center" paddingTop="10px">
    //             <Stack
    //                 direction="row"
    //                 spacing="10px"
    //                 alignItems="center"
    //                 justifyContent="center"
    //             >
    //                 {values.map((value, i) => {
    //                     return (
    //                         <InformativenessCircle
    //                             key={i}
    //                             value={value}
    //                             selected={value === props.value}
    //                         />
    //                     );
    //                 })}
    //             </Stack>
    //             <Typography variant="h6" fontWeight="normal">
    //                 {getText(props.value)}
    //             </Typography>
    //         </Stack>
    //     </Stack>
    // );
};

export default Informativeness;*/