import { MnistFrontendPage } from './app.po';

describe('mnist-frontend App', () => {
  let page: MnistFrontendPage;

  beforeEach(() => {
    page = new MnistFrontendPage();
  });

  it('should display message saying app works', () => {
    page.navigateTo();
    expect(page.getParagraphText()).toEqual('app works!');
  });
});
