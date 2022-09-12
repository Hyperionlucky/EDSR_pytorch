// prettier-ignore
// const myAxis = 
const data = [
  [[15442337, 0.4060, 35939927, 'RCAN', 2], [3521, 0.4977, 35939927, 'SRCNN', 2], [22121089, 0.4430, 1376048943, 'RDN', 2], [11046209, 0.3976, 35939927, 'Ours', 2], [40720385, 0.4471, 35939927, 'EDSR', 2]],
  [[15626977, 0.4702, 35939927, 'RCAN', 3], [0, 1.2879, 1376048943, 'Bicubic', 3], [22305729, 0.5742, 1376048943, 'RDN', 3], [3521, 1.1922, 35939927, 'SRCNN', 3], [11230849, 0.4670, 35939927, 'Ours', 3], [43670785, 0.5612, 35939927, 'EDSR', 3], ],
  [[15895110, 0.4755, 35939927, 'SAN', 3]],
  [[0, 0.5179, 1376048943, 'Bicubic', 2],[15710470, 0.4132, 35939927, 'SAN', 2]]];
option = {
legend: {
  show:false,
  // right: "-10px",
  top: -5,
  orient:'vertical',
  textStyle:{
    fontSize: 25,
    fontFamily: 'Times new roman',
  }
},
xAxis: {
  scale: true,
  // splitNumber: 5,
  name:'Parameters (K)',
  nameLocation: 'center',
  nameTextStyle: {
    fontWeight:'bolder',
    // fontStyle : 'oblique',
    fontFamily:'Times new roman',
    fontSize:25,
    padding:10
  },
  nameGap: 25,
  splitLine: {
    show: true,
    lineStyle:{
      type: 'dashed',
      width: 1,
      color:'rgb(0,0,0)'
    }
  },
  axisLabel:{
    fontSize: 20,
    fontFamily: 'Times new roman',
  }
},
yAxis: {
  scale: true,
  splitNumber: 5,
  name:'MAE (m)',
  nameLocation: 'center',
  nameTextStyle: {
    fontWeight:'bold',
    fontFamily:'Times new roman',
    fontSize:25,
    padding:10
  },
  nameGap: 35,
  splitLine: {
    show: true,
    lineStyle:{
      type: 'dashed',
      width: 1,
      color:'rgb(0,0,0)'
    }
  },
  axisLabel:{
    fontSize: 20,
    fontFamily: 'Times new roman',
  }
},
series: [
  {
    name: '10m-5m',
    data: data[0],
    type: 'scatter',
    symbolSize: function (data) {
      return 30;
    },
    emphasis: {
      focus: 'self'
    },
    labelLayout: {
      // x: 1000,
      // align: 'center',
      
      // fontFamily:'Times new roman',
      // hideOverlap: true,
      moveOverlap: 'shiftY',
      // draggable: true
      
    },
    itemStyle: {
      color: '#c24b5f',
      // borderType : 'solid',
      // opacity: 1
    },
    labelLine: {
      show: true,
      length2: 5,
      lineStyle: {
        color: '#bbb'
      },
      minTurnAngle: 30
    },
    label: {
      show: true,
      formatter: function (param) {
        return param.data[3];
      },
      minMargin: 20,
      position: ['150%', '120%'],
      fontWeight:'bold',
      fontSize: 20,
      fontFamily:'Times new roman',
      distance: 10
    }
  },
  {
    name: '30m-10m',
    data: data[1],
    type: 'scatter',
    symbolSize: function (data) {
      return 30;
    },
    emphasis: {
      focus: 'self'
    },
    labelLayout: {
      // y: 20,
      align: 'center',
      // hideOverlap: true,
      moveOverlap: 'shiftX',
    },
    itemStyle: {
      color: '#6bb6a5',
      // borderType : 'solid',
      // opacity: 1
    },
    labelLine: {
      show: true,
      length2: 5,
      lineStyle: {
        color: '#bbb'
      }
    },
    label: {
      show: true,
      formatter: function (param) {
        return param.data[3];
      },
      minMargin: 10,
      position: ['150%', '-120%'],
      fontWeight:'bold',
      fontSize: 20,
      // distance:10,
      fontFamily:'Times new roman',
    }
  },
  {
    name: '30m-10m',
    data: data[2],
    type: 'scatter',
    symbolSize: function (data) {
      return 30;
    },
    emphasis: {
      focus: 'self'
    },
    labelLayout: {
      // y: 20,
      align: 'center',
      // hideOverlap: true,
      moveOverlap: 'shiftX',
    },
    itemStyle: {
      color: '#6bb6a5',
      // borderType : 'solid',
      // opacity: 1
    },
    labelLine: {
      show: true,
      length2: 5,
      lineStyle: {
        color: '#bbb'
      }
    },
    label: {
      show: true,
      formatter: function (param) {
        return param.data[3];
      },
      minMargin: 10,
      position: 'right',
      
      fontWeight:'bold',
      fontSize: 20,
      fontFamily:'Times new roman',
      distance:25,
    }
  },
  {
    name: '30m-10m',
    data: data[3],
    type: 'scatter',
    symbolSize: function (data) {
      return 30;
    },
    emphasis: {
      focus: 'self'
    },
    itemStyle: {
      color: '#c24b5f',
      // borderType : 'solid',
      // opacity: 1
    },
    labelLine: {
      show: true,
      length2: 5,
      lineStyle: {
        color: '#bbb'
      }
    },
    label: {
      show: true,
      formatter: function (param) {
        return param.data[3];
      },
      minMargin: 10,
      position: 'right',
      fontWeight:'bold',
      fontSize: 20,
      fontFamily:'Times new roman',
      distance:15,
    }
  },
],
toolbox: {
  feature: {
    saveAsImage:{
      show: true,
      pixelRatio: 15,
      backgroundColor: '#fff'
    }
  }
  },
};


//https://echarts.apache.org/examples/zh/editor.html?c=scatter-label-align-top&code=PTAEAcCcFMBdYJbUgWgQcwHYHsYCgBjbTAZ1lABMBDWK0AXlAG09Q3mmBGAVgBZeATAGYhAdgA0oAAwA6XkO7dJCgJxCVKgRNAByAEoBhAIIA5HZIEBdSUymTZ3TqJWTOYgGxTeADhXzJOgBCCAQArgBGIeagVjYCApwJUr72MtxCgq4eXr7-unoAImYW1swKCam8KqLaquqa2joAyoYmxTGlXJxe7gJSLtIy6qLuytxqGloBAPKhkCTRscy8UqJ9Qt5Kg9xSimMTDQEAogUti508ot28onaD8jv79VO6Tabn1qzsTFzcve7VbSyG57UB1SaNQzvZSdO6cGQCbzOLIjHJ-IQBYJhSIEaJCTrxIS7NYDByiTKgNyonzogKFdr4mzlTiuGScTQCJ4QgItAxtPEXRJE7xVSruW5cw66WbzAU2eTi1abVKeUHgqU6E5nGE2Hi-RzdSqbO7ql7NaFgyxWgDceGw4EQxAYoAA3l8AB5Gd0IEgALldXzYJAIVAANtB_bBIKFoOJA6AQKASOBQwhYCZQgBbcLIf1KeOZhCYACSmFgyAAbmG81Ja3W4-xQJgqJmIzoAApUSAtuDIEigAAUAGkAJTmePN1sAGWwIcdmH9OgI0DLyHHjcn0AAKtB3bAmrAAJ7h_1uxtsABmxFgAHVoBgABawX06cLYUMUdfnq9lgBiLYQUNDxfLcEFbftMGgAB3UBIGwTMqEwL9Gx_fcEAALwjTh3HjABfBt2E3ABxKhwH9AR80bZNU1gKciwjANzyTB9sCgyNo1jeM2FTSCD2PCMzyY0Aj3ABidAobB4GgT8CKYqCEAoWAH39FkuPYIhQ1wF9IHQcIBzsAyxzU3C8K-fCvkPL0fVPeNgzDBioxjWSkxTNMM2zXNQEowiexfABZIwjkHTMx2czcZznBBiEXZdV0gZCmx7Hc9z4k9GO_a870fZ9X3fGS1NQ_9CyAkCwOgCDoNg-DEISy9ryaDCsJwxtzI3HsSLIsFvKDVzaPomymJIFi2OEjjnO4-jUoEtS2BEsSJKk_KhNAeTFOUylxvU98tJ0HS9IM8QpCMpiTJasyCJIZAkD9Zh40EtrW0XPoeFqygaCof1qFoWxPiYubFzsqT4s2khD2zd8Gsw_0L1CTACHnQcvqoEd0qYmBYDmTAwSkW0Ts26BM3AB8qBIazUYysIbp0S7QwvHRjM20MqBzUMpyoQ9sFCZ9ycbRNgJiOwZtAMMMAXXRYvLYGhcTB8FOgaYK2QJnOsczjlszbBFYVpXSIB2WL1gAANV7WvPNMCamgahI0rTdAAYgIAReHCbg6c2thEzfSAKGQLdD1E0AAffBTXo9sB7SoAg035zgGbUpmWboyCrcG4bobDS73dAcNMHQJSBDzLOeOgS2efPG3IEXO3whr-nltO89TcbBPoFDFPzyG1j2KcoWr0gBCgeh2H4airGB3ALsWxR-6hPRzGIEnzMZCRph8VxoSm_PQtMD8rt0CLFTBeW8BsFJ-dF1ge1Q9AVCsvQJ8XzfD868bvDnJnthNyeqR2VepHPvelwX655_q6EBpLV6oNwahkhgxGGcMEYDiRtPIWc9IBYyJOvFq-NCbE1JjdD-7AryUwBq3OmccmIt1ZuzTm3NCFh1APzPoWcRZYBiiuCBWcZZy21pAZW3c1ZCQ1lrRWfDdZgP1kbE2m1zaZlLvQ0AFcq7uBru4Kg3Br6e1wD7SAfsA5B1TEtISiYI5RyPCpCh54qFJwYgozuI0LwZ0EZQlcecHwFy8kXSaR40oKLYEo-2NdwgvxOpY5uzNW7tyomnUaPdlp9wHpLIeCDR6Dgnt2EKZdGxoKxuklsy9AFryFpvRs29d46QPpSI-QkT5n1HhfK-Wdb73nvjlJ-n5jKmTYJYPAuFrRAA