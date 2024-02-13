var layoutOptions = {
  name: 'cose-bilkent',
// 'draft', 'default' or 'proof"
  // - 'draft' fast cooling rate
  // - 'default' moderate cooling rate
  // - "proof" slow cooling rate
  quality: 'default',
  // Whether to include labels in node dimensions. Useful for avoiding label overlap
  nodeDimensionsIncludeLabels: true,
  // Node repulsion (non overlapping) multiplier
  nodeRepulsion: 10,
  // Ideal (intra-graph) edge length
  idealEdgeLength: 1000,
  // Divisor to compute edge forces
  edgeElasticity: 10,
  // Gravity force (constant)
  gravity: 5,
  // Maximum number of iterations to perform
  numIter: 10000,
  // Whether to tile disconnected nodes
  tile: true,
  // Type of layout animation. The option set is {'during', 'end', false}
  animate: 'end',
  // Duration for animate:end
  animationDuration: 2000,
  // Amount of vertical space to put between degree zero nodes during tiling (can also be a function)
  tilingPaddingVertical: 200,
  // Amount of horizontal space to put between degree zero nodes during tiling (can also be a function)
  tilingPaddingHorizontal: 200,
};

var cy = cytoscape({
  container: document.getElementById('cy'),

  style: cytoscape.stylesheet()
    .selector('node')
      .css({
        'width': 200,
        'height': 200,
        'content': '',
        'shape': 'ellipse',
        'background-color': '#909090',
        'color': '#383838',
        'text-opacity': 1,
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': 80,
        'text-halign' : 'center',
        'text-valign' : 'center',
        'min-zoomed-font-size': 12,
        'text-background-color' : '#ffffff',
        'text-background-shape' : 'roundrectangle',
        'text-background-opacity': .3,
        'text-background-padding' : 1,
      })
    .selector('node[node_name]')
      .css({
        'content': 'data(node_name)',
      })
    .selector('node[colour]')
      .css({
        'background-color': 'data(colour)',
      })
    .selector('node[node_size]')
      .css({
        'width': 'data(node_size)',
        'height': 'data(node_size)',
      })
    .selector('node[node_fontsize]')
      .css({
        'font-size': 'data(node_fontsize)',
      })
    .selector('edge')
      .css({
        'width': 50,
        'font-size': 80,
        'color': 'black',
        'content': '',
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': '#909090',
        'opacity': 0.8,
        'text-opacity': 1,
        'line-color': '#909090',
        'line-style': 'solid',
        'text-background-color' : '#ffffff',
        'text-background-shape' : 'roundrectangle',
        'text-background-opacity': 1,
        'text-background-padding' : 4,
        'min-zoomed-font-size': 10,
        'control-point-step-size': 'data(node_size)',
      })
    .selector('edge[edge_name]')
      .css({
        'content': 'data(edge_name)',
      })
    .selector('edge[edge_size]')
      .css({
        'width': 'data(edge_size)',
        'font-size': 'data(edge_size)',
      })
    .selector(':selected')
      .css({
        'background-color': 'black',
        'line-color': 'black',
        'target-arrow-color': 'black',
        'source-arrow-color': 'black',
        'opacity': 1
      })
    .selector('.faded')
      .css({
        'opacity': 0.25,
        'text-opacity': 0
      }),

  elements: {{elements}},

  layout: layoutOptions,

  ready: function(){
    window.cy = this;
  }
});

function addQtips() {
    cy.filter(function(ele, i, eles) {
        return ele.data('tooltip') !== undefined;
    } ).qtip({
        content: function(){
                var tooltip = '<div style="overflow: auto;"><span style="white-space:nowrap;">' + this.data('tooltip') + '</span></div>';
                return tooltip;
            },
        show: {event: 'mouseover'},
        hide: {event: 'mouseout'},
        style: {classes: 'qtip-bootstrap'},
        position: {at: 'center bottom'}
    });
}

cy.minZoom(.001);
cy.maxZoom(20);
addQtips();


function toImage(){
    document.getElementById("downloader").href = cy.jpg({ full: false, quality: 1.0, scale: 2}).replace(/^data:image\/[^;]/, 'data:application/octet-stream');
}

function saveAsSvg() {
    var svgContent = cy.svg({scale: 1, full: true});
    var blob = new Blob([svgContent], {type:"image/svg+xml;charset=utf-8"});
    document.getElementById("downloader_svg").href = URL.createObjectURL(blob);
}

function fit() {
    cy.fit();
}

function resetLayout() {
    cy.startBatch();
    cy.layout(layoutOptions).run();
    if (removed !== undefined) {
        removed.forEach(function( ele ) {
            if (ele.data('node_x') !== undefined) {
                ele.position('x', ele.data('node_x'));
                ele.position('y', ele.data('node_y'));
            }
        });
    }
    cy.endBatch();
}

var years = {{years}};
var slider = document.getElementById("myRange");
if (slider !== null) {
    var output = document.getElementById("demo");
    output.innerHTML = years[slider.value]; // Display the default slider value

    // Update the current slider value (each time you drag the slider handle)
    var removed = cy.collection();

    slider.oninput = function() {
        var mile = this.value;
        output.innerHTML = years[mile];

        cy.startBatch();
        removed.restore();
        removed = cy.remove("[mile<" + mile + "]");
        cy.endBatch();
    }
}

function toggleFullScreen() {
  var elem = document.getElementById("fullscreenDiv");
  if ((document.fullScreenElement && document.fullScreenElement !== null) ||
   (!document.mozFullScreen && !document.webkitIsFullScreen)) {
    if (elem.requestFullScreen) {
      elem.requestFullScreen();
    } else if (elem.mozRequestFullScreen) {
      elem.mozRequestFullScreen();
    } else if (elem.webkitRequestFullScreen) {
      elem.webkitRequestFullScreen(Element.ALLOW_KEYBOARD_INPUT);
    }
    document.getElementById("fsspan").className = "glyphicon glyphicon-resize-small"
    document.getElementById("fstext").innerHTML = "  Exit full screen"
  } else {
    if (document.cancelFullScreen) {
      document.cancelFullScreen();
    } else if (document.mozCancelFullScreen) {
      document.mozCancelFullScreen();
    } else if (document.webkitCancelFullScreen) {
      document.webkitCancelFullScreen();
    }
    document.getElementById("fsspan").className = "glyphicon glyphicon-fullscreen"
    document.getElementById("fstext").innerHTML = "  Full screen"
  }
}