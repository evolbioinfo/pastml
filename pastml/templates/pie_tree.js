
var layoutOptions = {
    name: '{{layout}}',
    nodesep: 10, // the separation between adjacent nodes in the same rank
    edgeSep: 10, // the separation between adjacent edges in the same rank
    rankSep: 80, // the separation between adjacent ranks
    rankDir: 'TB', // 'TB' for top to bottom flow, 'LR' for left to right,
    ranker: 'longest-path', // Type of algorithm to assign a rank to each node in the input graph. Possible values: 'network-simplex', 'tight-tree' or 'longest-path'
    minLen: 1, // number of ranks to keep between the source and target of the edge

    // general layout options
    fit: true, // whether to fit to viewport
    padding: 1, // fit padding
    spacingFactor: undefined, // Applies a multiplicative factor (>0) to expand or compress the overall area that the nodes take up
    nodeDimensionsIncludeLabels: true, // whether labels should be included in determining the space used by a node
    animate: false, // whether to transition the node positions
    animateFilter: function( node, i ){ return true; }, // whether to animate specific nodes when animation is on; non-animated nodes immediately go to their final positions
    animationDuration: 500, // duration of animation in ms if enabled
    animationEasing: undefined, // easing of animation if enabled
    boundingBox: undefined, // constrain layout bounds; { x1, y1, x2, y2 } or { x1, y1, w, h }
    transform: function( node, pos ){ return pos; }, // a function that applies a transform to the final node position
    ready: function(){}, // on layoutready
    stop: function(){} // on layoutstop
  };

var cy = cytoscape({
  container: document.getElementById('cy'),

  style: cytoscape.stylesheet()
    .selector('node')
      .css({
        'width': 300,
        'height': 300,
        'content': '',
        'shape': 'ellipse',
        'pie-size': '100%',
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
    .selector('node[tip]')
      .css({
        'width': 400,
        'height': 400,
        'font-size': 100,
      })
    .selector('node[unresolved]')
      .css({
        'shape': 'octagon',
        'pie-size': '92%',
        'width': 500,
        'height': 500,
        'font-size': 125,
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
    {% for (clazz, css) in clazz2css %}
    .selector(".{{clazz}}")
        .css({
        {{css}}
        })
    {% endfor %}
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
        'text-background-color' : '#ffffff',
        'text-background-shape' : 'roundrectangle',
        'text-background-opacity': 1,
        'text-background-padding' : 4,
        'min-zoomed-font-size': 10,
      })
    .selector('edge[edge_meta]')
      .css({
        'line-color': '#383838',
        'target-arrow-color': '#383838',
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
        'pie-size': '60%',
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
        return ele.isNode() && ele.data('tooltip') !== undefined;
    } ).qtip({
        content: function(){
                var tooltip = '<br><div style="overflow: auto;"><span style="white-space:nowrap;">' + this.data('tooltip') + '</span></div>';
                if (this.data('node_meta') !== undefined) {
                    tooltip += '<br><div style="overflow: auto;"><span style="white-space:nowrap;">ids: ' + this.data('node_names') + '</span></div>';
                } else {
                    tooltip += '<br>id: ' + this.data('node_names') + '<br>';
                }
                tooltip += '<br>tips inside: ' + this.data('node_in_tips');
                tooltip += '<br>total tips in the subtree: ' + this.data('node_all_tips');
                tooltip += '<br>internal nodes inside: ' + this.data('node_in_ns');
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

function fit() {
    cy.fit();
}

function resetLayout() {
    cy.layout(layoutOptions).run();
}

var years = {{years}};
var slider = document.getElementById("myRange");
if (slider !== null) {
    var output = document.getElementById("demo");
    output.innerHTML = years[slider.value]; // Display the default slider value

    // Update the current slider value (each time you drag the slider handle)
    var removed = cy.collection();

    slider.oninput = function() {
        output.innerHTML = years[this.value];
        removed.restore();
        removed = cy.remove("[date>" + this.value + "]");
        var list = cy.$("");
        for (var i=0, ele; ele = list[i]; i++) {
            if (ele.data('node_name_' + this.value) !== undefined) {
                ele.data('node_name', ele.data('node_name_' + this.value));
            }
            if (ele.data('node_names_' + this.value) !== undefined) {
                ele.data('node_names', ele.data('node_names_' + this.value));
            }
            if (ele.data('node_fontsize_' + this.value) !== undefined) {
                ele.data('node_fontsize', ele.data('node_fontsize_' + this.value));
            }
            if (ele.data('node_in_tips_' + this.value) !== undefined) {
                ele.data('node_in_tips', ele.data('node_in_tips_' + this.value));
            }
            if (ele.data('node_in_ns_' + this.value) !== undefined) {
                ele.data('node_in_ns', ele.data('node_in_ns_' + this.value));
            }
            if (ele.data('node_all_tips_' + this.value) !== undefined) {
                ele.data('node_all_tips', ele.data('node_all_tips_' + this.value));
            }
            if (ele.data('node_size_' + this.value) !== undefined) {
                ele.data('node_size', ele.data('node_size_' + this.value));
            }
            if (ele.data('edge_name_' + this.value) !== undefined) {
                ele.data('edge_name', ele.data('edge_name_' + this.value));
            }
            if (ele.data('edge_size_' + this.value) !== undefined) {
                ele.data('edge_size', ele.data('edge_size_' + this.value));
            }
            if (ele.data('edge_meta_' + this.value) !== undefined) {
                ele.data('edge_meta', ele.data('edge_meta_' + this.value))
            } else if (ele.data('edge_meta') !== undefined) {
                ele.removeData('edge_meta');
                cy.remove(ele);
                cy.add(ele);
            }
            if (ele.data('node_meta_' + this.value) !== undefined) {
                ele.data('node_meta', ele.data('node_meta_' + this.value));
            } else if (ele.data('node_meta') !== undefined) {
                ele.removeData('node_meta');
            }
        }
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