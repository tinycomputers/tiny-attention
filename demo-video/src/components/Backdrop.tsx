import React from "react";
import { AbsoluteFill } from "remotion";
import { C } from "../theme";

// Flat product background (the app's own base colour). No gradients, grids or grain.
export const Backdrop: React.FC = () => <AbsoluteFill style={{ background: C.bg }} />;
