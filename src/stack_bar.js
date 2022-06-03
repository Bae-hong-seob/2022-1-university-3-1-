var margin = {
    top: 30,
    right: 10,
    bottom: 30,
    left: 40
};


var width = 1100;
var height = 350;

// create the svg
var stack_svg = d3.select(".stack_bar")
.append("svg")
.attr("width", width + margin.left + margin.right)
.attr("height", height + margin.top + margin.bottom)

var g = stack_svg.append("g")
.attr("transform",
      "translate(" + margin.left + "," + margin.top + ")");



// set x scale
var x = d3.scaleBand()
    .rangeRound([0, width])
    .paddingInner(0.3)
    .align(0.1);

// set y scale
var y = d3.scaleLinear()
    .rangeRound([height, 0]);

// set the colors
var z = d3.scaleOrdinal()
    .range(["#3399ff", "#ff0033", "#66ff00", "#ffff33"]);

// load the csv and create the chart
d3.csv("/data/bar_total.csv", function(d, i, columns) {
  for (i = 1, t = 0; i < columns.length; ++i) t +=d[columns[i]] = parseInt(+d[columns[i]]);
  d.total = t;
  return d;
}, function(error, data) {
  if (error) throw error;

  var keys = data.columns.slice(1);

  data.sort(function(a, b) { return b.total - a.total; });
  x.domain(data.map(function(d) { return d.자치구; }));
  y.domain([0, 70]).nice();
  z.domain(keys);

  g.append("g")
    .selectAll("g")
    .data(d3.stack().keys(keys)(data))
    .enter().append("g")
      .attr("fill", function(d) { return z(d.key); })
    .selectAll("rect")
    .data(function(d) { return d; })
    .enter().append("rect")
      .attr("x", function(d) { return x(d.data.자치구); })
      .attr("y", function(d) { return y(d[1]); })
      .attr("height", function(d) { return y(d[0]) - y(d[1]); })
      .attr("width", x.bandwidth())
      .attr("stroke", "black")
      .attr("opacity", "0.8")
  
    .on("mousemove", function(d) {
      var xPosition = d3.mouse(this)[0] - 10;
      var yPosition = d3.mouse(this)[1] - 10;
      tooltip.attr("transform", "translate(" + xPosition + "," + yPosition + ")");
      tooltip.select("text").text(d[1]-d[0] + "점");
    })

    .on("mouseover", function (d) { 
      d3.select(this)
      g.append('line')
      .transition()
      .duration(300)
      .attr('x1', 0)
      .attr('y1', y(d[1]))
      .attr('x2', 1090)
      .attr('y2', y(d[1]))
      .attr('stroke', 'red')
      .attr('stroke-dasharray','2')
    
      tooltip.style("display", null)
    })

    .on("mouseout", function (d) { 
      d3.selectAll(".stack_bar").selectAll("line").style("display", "none")
      tooltip.style("display", "none")
});

  g.append("g")
      .attr("class", "axis")
      .attr("transform", "translate(0,350)")
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("transform", "translate(15,0)rotate(0)")
      .style("text-anchor", "end")
      .style("font-weight", "bold")
      .style("font-size", "11px");

  g.append("g")
      .attr("class", "axis")
      .call(d3.axisLeft(y).ticks(null, "s"))
    .append("text")
      .attr("x", 2)
      .attr("y", y(y.ticks().pop()) + 0.5)
      .attr("dy", "0.32em")
      .attr("fill", "#000")
      .attr("font-weight", "bold")
      .attr("text-anchor", "start");

  var legend = g.append("g")
    .attr("class", "legend_item")
    .attr("font-size", 13)
    .attr("text-anchor", "end")
    .selectAll("g")
    .data(keys.slice().reverse())
    .enter().append("g")
    .attr("transform", function(d, i) { return "translate(550," + i * 20 + ")"; });

  legend.append("rect")
      .attr("x", width - 19)
      .attr("width", 19)
      .attr("height", 19)
      .attr("fill", z)

  legend.append("text")
      .attr("x", width - 24)
      .attr("y", 9.5)
      .attr("dy", "0.32em")
      .text(function(d) { return d; });
});

  // Prep the tooltip bits, initial display is hidden
  var tooltip = stack_svg.append("g")
    .attr("class", "tooltip")
    .style("display", "none");
      
  tooltip.append("rect")
    .attr("width", 60)
    .attr("height", 35)
    .attr("fill", "white")
    .style("opacity", 0.5);

  tooltip.append("text")
    .attr("x", 30)
    .attr("dy", "1.2em")
    .style("text-anchor", "middle")
    .attr("font-size", "20px")
    .attr("font-weight", "bold");