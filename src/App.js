import logo from './logo.svg';
import './App.css';
import * as d3 from "d3";
import { useEffect, useRef, useState } from "react";
import DataVariable from './1_DataVariable';
import Line from './3_Line';
import Line_teacher from './3_Line_teacher';
import Bar from './4_Bar';
import Geojson from './5_Geojson';
import input_data from "./Report.csv";

function App() {
  let table = [];
  const readCsv = async () => {

    let file = await d3.csv(input_data);
    let valueKey = file.columns;
    file.map((load_csv,index) => table.push(parseInt(load_csv.num_school)))
    console.log(table)
  };

  useEffect(() => {
    readCsv();
  },);

  return (
    <>
      <DataVariable />
      <Line y_label={table}/>
      <Line_teacher/>
      <Bar/>
      <Geojson/>
    </>
  );
}

export default App;
