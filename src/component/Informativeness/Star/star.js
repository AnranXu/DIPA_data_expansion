import React from "react";

import StarIcon from "@mui/icons-material/Star";
import StarBorderIcon from "@mui/icons-material/StarBorder";

// props
// filled: boolean
const InformativenessStar = (props) => {
    return props.filled ? (
        <StarIcon fontSize="large" />
    ) : (
        <StarBorderIcon fontSize="large" />
    );
};

export default InformativenessStar;