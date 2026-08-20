// React and htm arrive as UMD globals from index.html, which is what lets this
// app run with no build step. Re-exporting them here keeps every component
// module free of window lookups.
const React = window.React;
const ReactDOM = window.ReactDOM;

export const html = window.htm.bind(React.createElement);
export const { useState, useEffect, useMemo, useRef, useCallback } = React;
export { React, ReactDOM };
