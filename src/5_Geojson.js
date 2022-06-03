import * as d3 from "d3";
import { useEffect, useRef, useState } from "react";
import input_data from "./Report.csv";

function Geojson() {
  const readCsv = async () => {

    // let column = columns;
    // console.log(column)

    let file = await d3.csv(input_data);
    // let valueKey = file.columns;
    // let table = [];
    // file.map((num_school,index) => table.push(num_school.year))
    // console.log(table)

      d3.select(svgRef.current)
      .selectAll("div.cities")
      .data(file)
      .enter()
      .append("div")
      .attr("class", "cities")
      .html((d, i) => d.year);
  };

  useEffect(() => {
    readCsv();
  },);

  const svgRef = useRef();
  return (
    <>
      <div ref={svgRef}></div>
    </>
  );
}

export default Geojson;