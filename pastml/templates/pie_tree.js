var layoutOptions = {
  name: 'preset',
  positions: function(node){
    node.position('x', node.data('node_x'));
    node.position('y', node.data('node_y'));
    return node.position();
  },
  fit: true, // whether to fit to viewport
  padding: 1, // padding on fit
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
    .selector('edge[edge_name>1]')
      .css({
        'line-color': '#383838',
        'target-arrow-color': '#383838',
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
                    tooltip += '<br><div style="overflow: auto;"><span style="white-space:nowrap;">ids: ' + this.data('node_roots') + '</span></div>';
                } else {
                    tooltip += '<br>id: ' + this.data('node_roots') + '<br>';
                }
                tooltip += '<br>{{tips}} inside: ' + this.data('node_in_tips');
                tooltip += '<br>total {{tips}} in the subtree: ' + this.data('node_all_tips');
                tooltip += '<br>{{internal_nodes}} inside: ' + this.data('node_in_ns');
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
        removed = cy.remove("[mile>" + mile + "]");
        cy.$("").forEach(function( ele ) {
            if (ele.data('node_name_' + mile) !== undefined) {
                ele.data('node_name', ele.data('node_name_' + mile));
            }
            if (ele.data('node_roots_' + mile) !== undefined) {
                ele.data('node_roots', ele.data('node_roots_' + mile));
            }
            if (ele.data('node_fontsize_' + mile) !== undefined) {
                ele.data('node_fontsize', ele.data('node_fontsize_' + mile));
            }
            if (ele.data('node_in_tips_' + mile) !== undefined) {
                ele.data('node_in_tips', ele.data('node_in_tips_' + mile));
            }
            if (ele.data('node_in_ns_' + mile) !== undefined) {
                ele.data('node_in_ns', ele.data('node_in_ns_' + mile));
            }
            if (ele.data('node_all_tips_' + mile) !== undefined) {
                ele.data('node_all_tips', ele.data('node_all_tips_' + mile));
            }
            if (ele.data('node_size_' + mile) !== undefined) {
                ele.data('node_size', ele.data('node_size_' + mile));
            }
            if (ele.data('edge_name_' + mile) !== undefined) {
                ele.data('edge_name', ele.data('edge_name_' + mile));
            }
            if (ele.data('edge_size_' + mile) !== undefined) {
                ele.data('edge_size', ele.data('edge_size_' + mile));
            }
            if (ele.data('node_meta_' + mile) !== undefined) {
                ele.data('node_meta', ele.data('node_meta_' + mile));
            } else if (ele.data('node_meta') !== undefined) {
                ele.removeData('node_meta');
            }
        });
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